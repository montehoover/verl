import ast
import io
import contextlib
import logging
from typing import Any, Dict, Set, Optional, Tuple


__all__ = ["run_python_snippet"]


# Configure module-level logger with a clear, human-readable format.
logger = logging.getLogger("snippet_runner")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)


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
    logger.info("Validating snippet AST for safety.")
    visitor = _SafetyVisitor()
    visitor.visit(tree)
    logger.info("Validation successful: no forbidden constructs detected.")


def _format_block(title: str, text: str, max_len: int = 1000) -> str:
    """Format a multi-line text block for clear logging with optional truncation."""
    if len(text) > max_len:
        text = text[:max_len] + "\n...[truncated]..."
    indented = "\n".join(f"    {line}" for line in text.splitlines() or [""])
    return f"{title}:\n{indented}"


def _prepare_snippet(snippet_string: str) -> Tuple[Optional[object], Optional[object]]:
    """
    Parse, validate, and compile the snippet into execution units.

    Returns:
        (pre_code, expr_code)
    Raises:
        ValueError on invalid or forbidden code.
    """
    if not isinstance(snippet_string, str):
        raise ValueError("Snippet must be a string.")

    logger.info(_format_block("Received code snippet", snippet_string))

    try:
        logger.info("Parsing snippet into AST.")
        tree = ast.parse(snippet_string, mode="exec")
    except SyntaxError as e:
        logger.error("Parsing failed due to syntax error: %s", e)
        raise ValueError(f"Invalid code: {e}") from None

    # Validate AST for safety
    _validate_snippet_ast(tree)

    # Determine if the last statement is an expression to evaluate
    last_expr = None
    pre_body = tree.body
    has_trailing_expr = bool(tree.body and isinstance(tree.body[-1], ast.Expr))
    if has_trailing_expr:
        last_expr = tree.body[-1].value
        pre_body = tree.body[:-1]
        logger.info("Detected trailing expression for evaluation.")
    else:
        logger.info("No trailing expression detected; statements-only snippet.")

    # Compile code blocks
    pre_code = None
    if pre_body:
        logger.info("Compiling statements block (%d statement%s).", len(pre_body), "" if len(pre_body) == 1 else "s")
        pre_mod = ast.Module(body=pre_body, type_ignores=[])
        pre_code = compile(pre_mod, filename="<snippet>", mode="exec")
    else:
        logger.info("No statements to execute before evaluation.")

    expr_code = None
    if last_expr is not None:
        logger.info("Compiling final expression.")
        expr_code = compile(ast.Expression(last_expr), filename="<snippet>", mode="eval")
    else:
        logger.info("No expression to evaluate.")

    logger.info("Preparation complete.")
    return pre_code, expr_code


def _execute_snippet(pre_code: Optional[object], expr_code: Optional[object], safe_builtins: Dict[str, Any]) -> Tuple[str, Any]:
    """
    Execute compiled code objects in a restricted environment and capture stdout.

    Returns:
        (printed_output, result_value)
    Raises:
        Exception from execution; callers should wrap and re-raise as needed.
    """
    logger.info("Initializing restricted execution environment.")
    safe_globals: Dict[str, Any] = {"__builtins__": safe_builtins.copy()}
    safe_locals: Dict[str, Any] = {}

    buf = io.StringIO()
    result: Any = None

    try:
        with contextlib.redirect_stdout(buf):
            if pre_code is not None:
                logger.info("Executing statements block.")
                exec(pre_code, safe_globals, safe_locals)
                logger.info("Statements block executed successfully.")
            if expr_code is not None:
                logger.info("Evaluating final expression.")
                result = eval(expr_code, safe_globals, safe_locals)
                logger.info("Final expression evaluated successfully. Result: %r", result)
    finally:
        printed_output = buf.getvalue()
        if printed_output:
            logger.info(_format_block("Captured printed output", printed_output, max_len=2000))
        else:
            logger.info("No printed output captured.")

    return printed_output, result


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
    try:
        pre_code, expr_code = _prepare_snippet(snippet_string)
    except ValueError:
        # Already a ValueError with a user-friendly message
        logger.error("Snippet preparation failed.")
        raise
    except Exception as e:
        logger.exception("Unexpected error during snippet preparation.")
        raise ValueError(f"Invalid code: {e}") from None

    try:
        printed_output, result = _execute_snippet(pre_code, expr_code, SAFE_BUILTINS)
    except Exception as e:
        logger.exception("Error during snippet execution.")
        raise ValueError(f"Error during execution: {e}") from None

    if printed_output:
        logger.info("Returning captured printed output (%d chars).", len(printed_output))
        return printed_output

    logger.info("Returning final result value: %r", result)
    return result
