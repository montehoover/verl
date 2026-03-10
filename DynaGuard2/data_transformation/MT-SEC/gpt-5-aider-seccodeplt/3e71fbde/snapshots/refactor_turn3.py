import ast
import io
import itertools
import logging
from contextlib import redirect_stdout

# Configure logging (only if no handlers are configured by host application)
logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

# Disallowed names when loaded (read/used). They may be used as targets (assigned) safely.
DISALLOWED_NAMES = {
    "__import__",
    "eval",
    "exec",
    "compile",
    "open",
    "input",
    "help",
    "exit",
    "quit",
    "globals",
    "locals",
    "vars",
    "dir",
    "getattr",
    "setattr",
    "delattr",
    "memoryview",
    "bytearray",
    "super",
    "type",
    "object",
}

# Calls to these names are disallowed as well.
DISALLOWED_CALLS = DISALLOWED_NAMES

# Disallowed AST node types for safety.
DISALLOWED_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.With,
    ast.AsyncWith,
    ast.AsyncFunctionDef,
    ast.Lambda,
    ast.ClassDef,
    ast.Global,
    ast.Nonlocal,
    ast.Try,
    ast.Raise,
    ast.Delete,
    ast.Await,
    ast.Yield,
    ast.YieldFrom,
)

SAFE_BUILTINS = {
    # Basic types and constructors
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "complex": complex,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    # Utilities
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "round": round,
    "sorted": sorted,
    "any": any,
    "all": all,
    "zip": zip,
    "map": map,
    "filter": filter,
    "enumerate": enumerate,
    "range": range,
    "pow": pow,
    "chr": chr,
    "ord": ord,
    "bin": bin,
    "hex": hex,
    "oct": oct,
    "reversed": reversed,
    # Printing (captured)
    "print": print,
}


class _SafetyChecker(ast.NodeVisitor):
    def visit(self, node):
        # Disallow certain node types outright
        if isinstance(node, DISALLOWED_NODES):
            raise ValueError(f"Disallowed operation: use of {type(node).__name__}")
        return super().visit(node)

    def visit_Import(self, node):
        raise ValueError("Disallowed operation: import statements are not allowed")

    def visit_ImportFrom(self, node):
        raise ValueError("Disallowed operation: import statements are not allowed")

    def visit_Attribute(self, node: ast.Attribute):
        # Disallow access to dunder attributes (e.g., __class__, __subclasses__, etc.)
        if isinstance(node.attr, str) and node.attr.startswith("__"):
            raise ValueError("Disallowed operation: access to dunder attributes is not allowed")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        # Disallow loading certain dangerous names; allow storing to them (shadowing) though
        if isinstance(node.ctx, ast.Load) and node.id in DISALLOWED_NAMES:
            raise ValueError(f"Disallowed operation: use of '{node.id}' is not allowed")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # Disallow calling of certain dangerous names
        func = node.func
        if isinstance(func, ast.Name) and func.id in DISALLOWED_CALLS:
            raise ValueError(f"Disallowed operation: calling '{func.id}' is not allowed")
        if isinstance(func, ast.Attribute) and isinstance(func.attr, str) and func.attr.startswith("__"):
            raise ValueError("Disallowed operation: calling dunder attributes is not allowed")
        self.generic_visit(node)


def _is_single_expression(tree: ast.AST) -> bool:
    return (
        isinstance(tree, ast.Module)
        and len(tree.body) == 1
        and isinstance(tree.body[0], ast.Expr)
    )


def parse_and_validate(snippet: str) -> tuple[ast.AST, bool]:
    """
    Parse the snippet into an AST and validate it with safety rules.
    Returns a tuple of (tree, is_single_expr).
    """
    try:
        tree = ast.parse(snippet, mode="exec")
    except SyntaxError:
        # Let SyntaxError bubble up to caller
        raise
    _SafetyChecker().visit(tree)
    return tree, _is_single_expression(tree)


def execute_parsed(tree: ast.AST, is_single_expr: bool) -> tuple[str, object]:
    """
    Execute previously parsed and validated AST in a restricted environment.
    Returns a tuple of (printed_output, value).
    """
    # Prepare a restricted execution environment
    exec_globals = {"__builtins__": SAFE_BUILTINS.copy()}
    exec_locals = {}

    stdout_buf = io.StringIO()
    with redirect_stdout(stdout_buf):
        if is_single_expr:
            # Evaluate the expression safely
            expr_node = ast.Expression(body=tree.body[0].value)  # type: ignore[attr-defined]
            compiled_expr = compile(expr_node, filename="<snippet>", mode="eval")
            value = eval(compiled_expr, exec_globals, exec_locals)
        else:
            compiled = compile(tree, filename="<snippet>", mode="exec")
            exec(compiled, exec_globals, exec_locals)
            value = None

    printed = stdout_buf.getvalue()
    return printed, value


_EXEC_COUNTER = itertools.count(1)


def _preview(text: str, max_len: int = 500) -> str:
    try:
        s = text.replace("\n", "\\n")
    except Exception:
        s = str(text)
    if len(s) > max_len:
        return s[:max_len] + "...[truncated]"
    return s


def run_code_snippet(snippet: str):
    """
    Execute a Python code snippet safely with strict rules.

    - snippet: str, Python code to execute.
    - Returns:
        * If the snippet produces printed output, return that output (as a string).
        * Else if the snippet is a single expression, return its evaluated value.
        * Else return None.
    - Raises:
        * ValueError if disallowed operations are detected.
    """
    if not isinstance(snippet, str):
        logger.error("run_code_snippet called with non-string snippet: type=%s", type(snippet).__name__)
        raise TypeError("snippet must be a string")

    exec_id = next(_EXEC_COUNTER)
    logger.info("Execution %d: starting. Snippet=%r", exec_id, _preview(snippet))

    try:
        tree, is_single_expr = parse_and_validate(snippet)
        printed, value = execute_parsed(tree, is_single_expr)
    except Exception as e:
        logger.exception("Execution %d: failed with %s: %s", exec_id, type(e).__name__, e)
        raise

    if printed.strip():
        logger.info(
            "Execution %d: success. Outcome=stdout length=%d preview=%r",
            exec_id,
            len(printed),
            _preview(printed),
        )
        return printed

    if value is None:
        logger.info("Execution %d: success. Outcome=return_none", exec_id)
        return None

    try:
        value_repr = repr(value)
    except Exception:
        value_repr = f"<unrepr-able {type(value).__name__}>"

    logger.info(
        "Execution %d: success. Outcome=expr_value preview=%r",
        exec_id,
        _preview(value_repr),
    )
    return value
