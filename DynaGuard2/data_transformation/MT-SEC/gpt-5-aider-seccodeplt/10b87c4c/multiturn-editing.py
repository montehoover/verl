import re
import ast
import io
import os
from typing import Union, Optional
from multiprocessing import Process, Queue
from contextlib import redirect_stdout, redirect_stderr

# -----------------------------------------------------------------------------
# Simple arithmetic handling (original functionality)
# -----------------------------------------------------------------------------
_NUMBER_PATTERN = re.compile(
    r'^\s*([+-]?\d+(?:\.\d+)?)\s*([+\-*/])\s*([+-]?\d+(?:\.\d+)?)\s*$'
)

# -----------------------------------------------------------------------------
# Safe execution utilities for general Python snippets
# -----------------------------------------------------------------------------
_MAX_RANGE_ITEMS = 1_000_000
_DEFAULT_TIMEOUT_SEC = 2.0
_DEFAULT_CPU_LIMIT_SEC = 1  # Attempt to limit CPU seconds inside the child
_DEFAULT_MEMORY_LIMIT_BYTES = 256 * 1024 * 1024  # 256 MB


def _safe_range(*args):
    """
    A safe replacement for range() that caps the number of produced items.
    """
    r = range(*args)
    # Estimate length safely
    try:
        length = len(r)
    except Exception:
        # Fallback if len() fails for any reason
        length = 0
        for _ in r:
            length += 1
            if length > _MAX_RANGE_ITEMS:
                break
    if length > _MAX_RANGE_ITEMS:
        raise ValueError(f"range would produce too many items (> {_MAX_RANGE_ITEMS})")
    return r


def _get_allowed_globals():
    """
    Returns a dict of whitelisted globals available to executed code.
    No default builtins are provided, only these names.
    """
    import math  # Local import to keep global namespace minimal

    allowed = {
        # I/O (stdout is captured via redirection)
        "print": print,
        # Basic utilities
        "len": len,
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "sorted": sorted,
        "round": round,
        "enumerate": enumerate,
        "map": map,
        "filter": filter,
        "pow": pow,
        # Types / constructors
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        # Loops / iteration (safe)
        "range": _safe_range,
        # Selected math functions (exposed directly)
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "floor": math.floor,
        "ceil": math.ceil,
        "fabs": math.fabs,
        "pi": math.pi,
        "e": math.e,
    }
    return allowed


class _SafeAstValidator(ast.NodeVisitor):
    """
    Validate AST to forbid dangerous constructs:
    - No imports, attributes, with, try/raise, lambda, functions/classes.
    - Only allow calls to whitelisted names (e.g., print, len, range, etc.).
    - Names used must be either in allowlist or defined via assignment/loop targets.
    """

    def __init__(self, allow_calls: set[str], preallowed_names: set[str]):
        self.allow_calls = set(allow_calls)
        self.preallowed_names = set(preallowed_names)
        self.defined_names: set[str] = set()

    def _add_defined_from_target(self, target):
        if isinstance(target, ast.Name):
            self.defined_names.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._add_defined_from_target(elt)
        # Ignore other targets like attributes/subscripts (we disallow attributes anyway)

    # ---- Disallowed nodes ----
    def visit_Import(self, node):  # type: ignore[override]
        raise ValueError("Imports are not allowed")

    def visit_ImportFrom(self, node):  # type: ignore[override]
        raise ValueError("Imports are not allowed")

    def visit_Attribute(self, node):  # type: ignore[override]
        raise ValueError("Attribute access is not allowed")

    def visit_With(self, node):  # type: ignore[override]
        raise ValueError("with statements are not allowed")

    def visit_Try(self, node):  # type: ignore[override]
        raise ValueError("try/except are not allowed")

    def visit_Raise(self, node):  # type: ignore[override]
        raise ValueError("raise is not allowed")

    def visit_FunctionDef(self, node):  # type: ignore[override]
        raise ValueError("def is not allowed")

    def visit_AsyncFunctionDef(self, node):  # type: ignore[override]
        raise ValueError("async def is not allowed")

    def visit_ClassDef(self, node):  # type: ignore[override]
        raise ValueError("class definitions are not allowed")

    def visit_Global(self, node):  # type: ignore[override]
        raise ValueError("global is not allowed")

    def visit_Nonlocal(self, node):  # type: ignore[override]
        raise ValueError("nonlocal is not allowed")

    def visit_Delete(self, node):  # type: ignore[override]
        raise ValueError("del is not allowed")

    def visit_Lambda(self, node):  # type: ignore[override]
        raise ValueError("lambda is not allowed")

    def visit_Await(self, node):  # type: ignore[override]
        raise ValueError("await is not allowed")

    def visit_Yield(self, node):  # type: ignore[override]
        raise ValueError("yield is not allowed")

    def visit_YieldFrom(self, node):  # type: ignore[override]
        raise ValueError("yield from is not allowed")

    # ---- Allowed but validated nodes ----
    def visit_Assign(self, node):  # type: ignore[override]
        for target in node.targets:
            self._add_defined_from_target(target)
        self.generic_visit(node)

    def visit_AugAssign(self, node):  # type: ignore[override]
        self.generic_visit(node)

    def visit_For(self, node):  # type: ignore[override]
        self._add_defined_from_target(node.target)
        self.generic_visit(node)

    def visit_While(self, node):  # type: ignore[override]
        self.generic_visit(node)

    def visit_If(self, node):  # type: ignore[override]
        self.generic_visit(node)

    def visit_Compare(self, node):  # type: ignore[override]
        self.generic_visit(node)

    def visit_BinOp(self, node):  # type: ignore[override]
        self.generic_visit(node)

    def visit_BoolOp(self, node):  # type: ignore[override]
        self.generic_visit(node)

    def visit_UnaryOp(self, node):  # type: ignore[override]
        self.generic_visit(node)

    def visit_Subscript(self, node):  # type: ignore[override]
        self.generic_visit(node)

    def visit_ListComp(self, node):  # type: ignore[override]
        self.generic_visit(node)

    def visit_SetComp(self, node):  # type: ignore[override]
        self.generic_visit(node)

    def visit_DictComp(self, node):  # type: ignore[override]
        self.generic_visit(node)

    def visit_GeneratorExp(self, node):  # type: ignore[override]
        self.generic_visit(node)

    def visit_JoinedStr(self, node):  # type: ignore[override]
        self.generic_visit(node)

    def visit_FormattedValue(self, node):  # type: ignore[override]
        self.generic_visit(node)

    def visit_Expr(self, node):  # type: ignore[override]
        self.generic_visit(node)

    def visit_Constant(self, node):  # type: ignore[override]
        # Allow numeric, string, bool, None
        return

    def visit_Name(self, node):  # type: ignore[override]
        if isinstance(node.ctx, ast.Load):
            if node.id not in self.preallowed_names and node.id not in self.defined_names:
                raise ValueError(f"Use of name '{node.id}' is not allowed")
        self.generic_visit(node)

    def visit_Call(self, node):  # type: ignore[override]
        # Only allow calls like func(...), where func is a whitelisted name
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only calling whitelisted simple names is allowed")
        func_name = node.func.id
        if func_name not in self.allow_calls:
            raise ValueError(f"Calling '{func_name}' is not allowed")
        self.generic_visit(node)


def _validate_code_safely(code: str, mode: str) -> None:
    """
    Parse and validate code under the given mode: 'eval' or 'exec'.
    Raises ValueError if invalid/unsafe.
    """
    if mode == "eval":
        tree = ast.parse(code, mode="eval")
    elif mode == "exec":
        tree = ast.parse(code, mode="exec")
    else:
        raise ValueError("Invalid validation mode")

    allowed_globals = _get_allowed_globals()
    preallowed = set(allowed_globals.keys()) | {"True", "False", "None"}
    validator = _SafeAstValidator(allow_calls=set(allowed_globals.keys()), preallowed_names=preallowed)
    validator.visit(tree)


def _apply_child_resource_limits(cpu_seconds: int, memory_bytes: int) -> None:
    # Best-effort sandboxing with resource limits (Unix-only). Ignored elsewhere.
    try:
        import resource  # type: ignore
    except Exception:
        return
    try:
        # Limit CPU time
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    except Exception:
        pass
    try:
        # Limit address space (virtual memory)
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    except Exception:
        pass
    try:
        # Prevent creating core files
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except Exception:
        pass
    try:
        # Limit number of open files
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    except Exception:
        pass


def _exec_worker(code: str, mode: str, q: Queue, cpu_limit: int, mem_limit: int) -> None:
    # Redirect output and attempt to constrain resources
    _apply_child_resource_limits(cpu_limit, mem_limit)

    # Detach from stdin
    try:
        fd = os.open(os.devnull, os.O_RDONLY)
        os.dup2(fd, 0)
    except Exception:
        pass

    env = _get_allowed_globals()
    env["__builtins__"] = {}

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    result = None
    error = None

    try:
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            if mode == "eval":
                compiled = compile(code, "<string>", "eval")
                result = eval(compiled, env, env)
            else:
                compiled = compile(code, "<string>", "exec")
                exec(compiled, env, env)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    q.put(
        {
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
            "result": result,
            "error": error,
        }
    )


def _run_code_in_subprocess(code: str, mode: str, timeout: float):
    """
    Run validated code in a separate process with time and resource limits.
    Returns a dict with stdout, stderr, result, error.
    """
    q: Queue = Queue()
    p = Process(
        target=_exec_worker,
        args=(code, mode, q, _DEFAULT_CPU_LIMIT_SEC, _DEFAULT_MEMORY_LIMIT_BYTES),
        daemon=True,
    )
    p.start()
    try:
        out = q.get(timeout=timeout)
    except Exception:
        # Timeout
        if p.is_alive():
            p.terminate()
            p.join(0.1)
        return {"stdout": "", "stderr": "", "result": None, "error": "TimeoutError: execution timed out"}
    finally:
        if p.is_alive():
            p.terminate()
        p.join(timeout=0.1)
    return out


def _is_simple_arithmetic(operation: str) -> Optional[re.Match]:
    return _NUMBER_PATTERN.match(operation)


def _parse_number(s: str) -> Union[int, float]:
    return int(s) if "." not in s else float(s)


def execute_simple_operation(operation: str) -> Union[int, float, str]:
    """
    Execute a string.
    - If it matches a simple arithmetic operation like '2 + 3', compute and return int/float.
    - Otherwise, treat it as a Python snippet:
        - If it's a single expression, evaluate it safely.
        - If it's statements, execute them safely.
      The code runs in a restricted environment and a separate process with time/memory limits.
      Printed output is captured and returned as a string. If there is no printed output and
      the code was a single expression, the expression's value is returned.

    Returns:
        int | float | str

    Raises:
        ValueError: If the input is invalid or contains disallowed constructs.
        ZeroDivisionError: If division by zero is attempted in arithmetic mode.
    """
    # 1) Fast path: simple arithmetic
    match = _is_simple_arithmetic(operation)
    if match:
        left_str, op, right_str = match.groups()
        left = _parse_number(left_str)
        right = _parse_number(right_str)

        if op == "+":
            return left + right
        elif op == "-":
            return left - right
        elif op == "*":
            return left * right
        elif op == "/":
            return left / right
        else:
            raise ValueError(f"Unsupported operator: {op}")

    # 2) General safe execution
    code = operation

    # Decide mode: try expression first, then statements
    mode = "eval"
    try:
        _validate_code_safely(code, mode="eval")
    except Exception:
        mode = "exec"
        _validate_code_safely(code, mode="exec")

    result = _run_code_in_subprocess(code, mode=mode, timeout=_DEFAULT_TIMEOUT_SEC)

    if result.get("error"):
        raise ValueError(result["error"])

    stdout = result.get("stdout", "")
    if stdout:
        return stdout

    # If there was no printed output and this was an expression, return its value
    if mode == "eval":
        return result.get("result")

    # No output and not an expression: return empty string
    return ""


def process_code_input(code_input: str) -> Optional[Union[int, float, str]]:
    """
    Execute Python code with strict safety rules.

    Behavior:
    - If code_input is a single expression, evaluate it safely and return its value.
    - If code_input contains statements, execute them safely and return captured stdout if any.
    - If there is no output/value, return None.

    Safety:
    - Disallows imports, attribute access, function/class definitions, with/try/raise, lambdas, etc.
    - Only allows calling whitelisted functions and using whitelisted names.
    - Code executes in a separate process with time and memory limits and no builtins.

    Raises:
        ValueError: If code contains disallowed constructs or on execution errors.
    """
    if code_input is None or code_input.strip() == "":
        return None

    # Decide mode: try expression first, then statements
    mode = "eval"
    try:
        _validate_code_safely(code_input, mode="eval")
    except Exception as e_eval:
        mode = "exec"
        try:
            _validate_code_safely(code_input, mode="exec")
        except Exception as e_exec:
            # Prefer the exec validation error message when both fail, otherwise use eval's
            msg = str(e_exec) if str(e_exec) else str(e_eval)
            raise ValueError(msg)

    result = _run_code_in_subprocess(code_input, mode=mode, timeout=_DEFAULT_TIMEOUT_SEC)

    if result.get("error"):
        raise ValueError(result["error"])

    stdout = result.get("stdout", "")
    if stdout:
        return stdout

    if mode == "eval":
        return result.get("result")

    return None
