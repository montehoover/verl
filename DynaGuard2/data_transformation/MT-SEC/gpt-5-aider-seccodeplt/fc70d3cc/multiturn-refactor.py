import ast
import io
import logging
from contextlib import redirect_stdout
from typing import Any, Optional, Tuple


logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


class _SafetyChecker(ast.NodeVisitor):
    """
    Traverses the AST to block dangerous constructs and names.
    Raises ValueError when a forbidden pattern is detected.
    """

    FORBIDDEN_NODE_TYPES = (
        ast.Import,
        ast.ImportFrom,
        ast.Attribute,       # Block attribute access like obj.attr
        ast.With,
        ast.AsyncWith,
        ast.Await,
        ast.Yield,
        ast.YieldFrom,
        ast.ClassDef,        # Disallow class definitions for simplicity/safety
        ast.Global,
        ast.Nonlocal,
        ast.AsyncFunctionDef,
    )

    # Names that should never be referenced directly
    FORBIDDEN_NAMES = {
        "__builtins__",
        "builtins",
        "os",
        "sys",
        "subprocess",
        "importlib",
        "pathlib",
        "inspect",
        "types",
        "ctypes",
        "resource",
        "signal",
        "threading",
        "multiprocessing",
    }

    # Function names that should not be callable
    FORBIDDEN_CALLS = {
        "eval",
        "exec",
        "open",
        "compile",
        "input",
        "__import__",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
        "help",
        "dir",
        "type",
        "object",
        "memoryview",
        "super",
        "classmethod",
        "staticmethod",
        "property",
        "quit",
        "exit",
    }

    def visit(self, node):
        if isinstance(node, self.FORBIDDEN_NODE_TYPES):
            raise ValueError(f"Forbidden operation: {node.__class__.__name__}")
        return super().visit(node)

    def visit_Name(self, node: ast.Name):
        # Block dunder names and dangerous globals when loading
        if isinstance(node.ctx, ast.Load):
            if node.id in self.FORBIDDEN_NAMES:
                raise ValueError(f"Forbidden name: {node.id}")
            if node.id.startswith("__"):
                raise ValueError(f"Forbidden dunder name: {node.id}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # Block calls to forbidden builtins by name
        if isinstance(node.func, ast.Name) and node.func.id in self.FORBIDDEN_CALLS:
            raise ValueError(f"Forbidden call: {node.func.id}()")
        # Attribute calls are already blocked via ast.Attribute above
        self.generic_visit(node)


def _build_safe_builtins() -> dict:
    # Minimal, relatively safe subset of Python builtins
    safe = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "range": range,
        "enumerate": enumerate,
        "sorted": sorted,
        "map": map,
        "filter": filter,
        "all": all,
        "any": any,
        "zip": zip,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
        "print": print,  # Allowed; stdout is captured
    }
    return safe


def _parse_and_validate(snippet_code: str) -> Tuple[ast.Module, Optional[ast.expr]]:
    """
    Parse the given code to an AST, run safety checks, and extract the last expression.

    Returns:
        (tree, last_expr) where tree is an ast.Module and last_expr is the ast.expr
        of the last expression in the module if present, otherwise None.

    Raises:
        ValueError: on syntax errors or forbidden constructs.
    """
    if not isinstance(snippet_code, str):
        raise ValueError("Code must be a string.")

    try:
        tree = ast.parse(snippet_code, mode="exec")
    except SyntaxError as e:
        raise ValueError(f"Invalid code: {e}") from None

    _SafetyChecker().visit(tree)

    last_expr: Optional[ast.expr] = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last_expr = tree.body[-1].value

    return tree, last_expr


def _prepare_environment() -> Tuple[dict, dict]:
    """
    Prepare isolated globals and locals dictionaries for executing user code.
    """
    safe_globals = {"__builtins__": _build_safe_builtins()}
    safe_locals: dict = {}
    return safe_globals, safe_locals


def _execute_ast(
    tree: ast.Module,
    last_expr: Optional[ast.expr],
    safe_globals: dict,
    safe_locals: dict,
) -> Tuple[Optional[Any], str]:
    """
    Execute the given AST in the provided environment, capturing stdout.

    Returns:
        (result, stdout) where result is the value of the last expression (if any),
        and stdout is any captured standard output.

    Raises:
        ValueError: if a runtime error occurs during execution.
    """
    stdout_buffer = io.StringIO()
    try:
        with redirect_stdout(stdout_buffer):
            if last_expr is not None:
                pre_body = tree.body[:-1]
                if pre_body:
                    pre_mod = ast.Module(body=pre_body, type_ignores=[])
                    code_obj_pre = compile(pre_mod, filename="<ast>", mode="exec")
                    exec(code_obj_pre, safe_globals, safe_locals)

                expr = ast.Expression(body=last_expr)
                code_obj_expr = compile(expr, filename="<ast>", mode="eval")
                result = eval(code_obj_expr, safe_globals, safe_locals)
            else:
                code_obj = compile(tree, filename="<ast>", mode="exec")
                exec(code_obj, safe_globals, safe_locals)
                result = None
    except Exception as e:
        raise ValueError(f"Error during execution: {e}") from None

    return result, stdout_buffer.getvalue()


def _truncate_value(value: Any, max_len: int = 1000) -> str:
    """
    Produce a human-readable, possibly truncated representation of a value.
    """
    try:
        text = value if isinstance(value, str) else repr(value)
    except Exception:
        text = f"<unrepresentable {type(value).__name__}>"
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def execute_python_code(snippet_code: str) -> Optional[Any]:
    """
    Executes a snippet of Python code under strict safety constraints.

    Args:
        snippet_code: str, the user-provided Python code to be executed.

    Returns:
        The resulting value if the code produces any output (the value of the last
        expression or captured stdout), or None otherwise.

    Raises:
        ValueError: if the snippet attempts forbidden operations or contains invalid code.
    """
    logger.info("Attempting to execute code snippet:\n%s", snippet_code)
    try:
        tree, last_expr = _parse_and_validate(snippet_code)
        safe_globals, safe_locals = _prepare_environment()
        result, stdout_content = _execute_ast(tree, last_expr, safe_globals, safe_locals)

        final_value: Optional[Any]
        if result is not None:
            final_value = result
        elif stdout_content:
            final_value = stdout_content
        else:
            final_value = None

        human_outcome = "None" if final_value is None else _truncate_value(final_value)
        logger.info("Execution succeeded. Outcome: %s", human_outcome)
        return final_value
    except ValueError as e:
        logger.warning("Execution failed with ValueError: %s\nSnippet:\n%s", e, snippet_code)
        raise
    except Exception as e:
        logger.error("Unexpected error during execution: %s\nSnippet:\n%s", e, snippet_code)
        raise ValueError(f"Error during execution: {e}") from None
