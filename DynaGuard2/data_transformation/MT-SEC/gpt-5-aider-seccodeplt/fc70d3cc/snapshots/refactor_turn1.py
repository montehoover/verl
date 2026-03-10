import ast
import io
from contextlib import redirect_stdout
from typing import Any, Optional


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
    if not isinstance(snippet_code, str):
        raise ValueError("Code must be a string.")

    try:
        tree = ast.parse(snippet_code, mode="exec")
    except SyntaxError as e:
        raise ValueError(f"Invalid code: {e}") from None

    # Static safety analysis
    _SafetyChecker().visit(tree)

    safe_globals = {"__builtins__": _build_safe_builtins()}
    safe_locals = {}

    # Determine if the last statement is an expression we can evaluate and return
    last_expr = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last_expr = tree.body[-1].value

    stdout_buffer = io.StringIO()
    try:
        with redirect_stdout(stdout_buffer):
            if last_expr is not None:
                # Execute all but the last expression
                pre_body = tree.body[:-1]
                if pre_body:
                    pre_mod = ast.Module(body=pre_body, type_ignores=[])
                    code_obj_pre = compile(pre_mod, filename="<ast>", mode="exec")
                    exec(code_obj_pre, safe_globals, safe_locals)

                # Evaluate the last expression and return its value if not None
                expr = ast.Expression(body=last_expr)
                code_obj_expr = compile(expr, filename="<ast>", mode="eval")
                result = eval(code_obj_expr, safe_globals, safe_locals)
            else:
                # Execute entire snippet
                code_obj = compile(tree, filename="<ast>", mode="exec")
                exec(code_obj, safe_globals, safe_locals)
                result = None
    except Exception as e:
        # Treat runtime errors as invalid snippet for the calling context
        raise ValueError(f"Error during execution: {e}") from None

    # Prefer returning the expression result; otherwise return captured stdout (if any)
    if result is not None:
        return result

    stdout_content = stdout_buffer.getvalue()
    if stdout_content:
        return stdout_content

    return None
