import ast
import io
import contextlib
from typing import Any, Dict, Set


__all__ = ["run_python_snippet"]


# Whitelisted builtins available to the snippet
SAFE_BUILTINS: Dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bin": bin,
    "bool": bool,
    "callable": callable,
    "chr": chr,
    "complex": complex,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "format": format,
    "frozenset": frozenset,
    "hash": hash,
    "hex": hex,
    "int": int,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "iter": iter,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "oct": oct,
    "ord": ord,
    "pow": pow,
    "print": print,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


# Names that are forbidden to reference or call
FORBIDDEN_NAMES: Set[str] = {
    # Dangerous builtins and helpers
    "open",
    "exec",
    "eval",
    "compile",
    "__import__",
    "input",
    "globals",
    "locals",
    "vars",
    "dir",
    "type",
    "object",
    "super",
    "getattr",
    "setattr",
    "delattr",
    "memoryview",
    "staticmethod",
    "classmethod",
    "property",
    # Common modules used for system access
    "os",
    "sys",
    "subprocess",
    "shutil",
    "pathlib",
    "ctypes",
    "importlib",
    "builtins",
    "__builtins__",
}


class _SafetyVisitor(ast.NodeVisitor):
    """AST visitor that rejects potentially dangerous constructs."""

    def visit_Import(self, node: ast.Import) -> None:
        raise ValueError("Forbidden operation: import statements are not allowed.")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        raise ValueError("Forbidden operation: import statements are not allowed.")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # Block attribute access to prevent escaping via dunder traversal, etc.
        raise ValueError("Forbidden operation: attribute access is not allowed.")

    def visit_With(self, node: ast.With) -> None:
        raise ValueError("Forbidden operation: with statements are not allowed.")

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        raise ValueError("Forbidden operation: async with is not allowed.")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        raise ValueError("Forbidden operation: async functions are not allowed.")

    def visit_Await(self, node: ast.Await) -> None:
        raise ValueError("Forbidden operation: await is not allowed.")

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # Disallow class creation to reduce attribute/magic method surface.
        raise ValueError("Forbidden operation: class definitions are not allowed.")

    def visit_Call(self, node: ast.Call) -> None:
        # Ensure the callable is a Name (not attribute or other complex object)
        func = node.func
        if not isinstance(func, ast.Name):
            raise ValueError("Forbidden operation: indirect or attribute-based calls are not allowed.")
        # The name referenced must not be forbidden
        if func.id in FORBIDDEN_NAMES or func.id.startswith("__"):
            raise ValueError(f"Forbidden operation: use of '{func.id}' is not allowed.")
        # Visit arguments to ensure they don't contain forbidden names/attributes
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        # Forbid dangerous names when loading them
        if isinstance(node.ctx, ast.Load):
            if node.id in FORBIDDEN_NAMES or node.id.startswith("__"):
                raise ValueError(f"Forbidden operation: use of '{node.id}' is not allowed.")


def _validate_snippet_ast(tree: ast.AST) -> None:
    visitor = _SafetyVisitor()
    visitor.visit(tree)


def run_python_snippet(snippet_string: str) -> Any:
    """
    Execute a Python code snippet safely with restricted capabilities.

    Arguments:
    - snippet_string: str - the user-provided Python code to be executed.

    Returns:
    - The resulting value if the code produces any output (captured stdout) or
      the value of the final expression. Returns None otherwise.

    Raises:
    - ValueError if the snippet attempts forbidden operations or contains invalid code.
    """
    if not isinstance(snippet_string, str):
        raise ValueError("Snippet must be a string.")

    try:
        tree = ast.parse(snippet_string, mode="exec")
    except SyntaxError as e:
        raise ValueError(f"Invalid code: {e}") from None

    # Validate AST for safety
    _validate_snippet_ast(tree)

    # Determine if the last statement is an expression to evaluate
    last_expr = None
    pre_body = tree.body
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last_expr = tree.body[-1].value
        pre_body = tree.body[:-1]

    # Prepare isolated execution environment with restricted builtins
    safe_globals: Dict[str, Any] = {"__builtins__": SAFE_BUILTINS.copy()}
    safe_locals: Dict[str, Any] = {}

    # Compile code blocks
    pre_code = None
    if pre_body:
        pre_mod = ast.Module(body=pre_body, type_ignores=[])
        pre_code = compile(pre_mod, filename="<snippet>", mode="exec")

    expr_code = None
    if last_expr is not None:
        expr_code = compile(ast.Expression(last_expr), filename="<snippet>", mode="eval")

    # Execute while capturing stdout
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            if pre_code is not None:
                exec(pre_code, safe_globals, safe_locals)
            if expr_code is not None:
                result = eval(expr_code, safe_globals, safe_locals)
            else:
                # If there is no final expression, execute the full snippet (already done if pre_code existed)
                result = None
                if pre_code is None:
                    # Nothing executed yet: execute the (possibly empty) tree to allow statements-only snippets
                    full_code = compile(tree, filename="<snippet>", mode="exec")
                    exec(full_code, safe_globals, safe_locals)
    except Exception as e:
        raise ValueError(f"Error during execution: {e}") from None

    printed_output = buf.getvalue()

    # Prefer returning any printed output; otherwise return final expression value if present.
    if printed_output:
        return printed_output
    return result
