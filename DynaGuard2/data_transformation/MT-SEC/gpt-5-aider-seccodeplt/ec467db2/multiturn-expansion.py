import ast
from typing import Any


def filter_unsafe_operations(script: str) -> bool:
    """
    Returns True if the script contains only safe operations:
    - Expression statements only (no assignments, imports, control flow, etc.)
    - Expressions limited to:
        * Constants: int, float, complex, str, bool
        * Names (Load context only)
        * Unary +/- on allowed expressions
        * Binary operations using +, -, *, /, //, %, ** on allowed expressions
    Disallows:
        - Calls, attributes, subscripts, comprehensions, f-strings, formatting
        - Any statements other than bare expressions
        - Any other operator types or expression forms
    """
    try:
        tree = ast.parse(script, mode="exec")
    except SyntaxError:
        return False

    allowed_binops = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
    allowed_unaryops = (ast.UAdd, ast.USub)

    def is_allowed_constant(value: Any) -> bool:
        # Allow numeric and string constants (bool is a subclass of int; still acceptable)
        return isinstance(value, (int, float, complex, str, bool))

    def is_safe_expr(node: ast.AST) -> bool:
        if isinstance(node, ast.Constant):
            return is_allowed_constant(node.value)

        if isinstance(node, ast.UnaryOp):
            return isinstance(node.op, allowed_unaryops) and is_safe_expr(node.operand)

        if isinstance(node, ast.BinOp):
            return isinstance(node.op, allowed_binops) and is_safe_expr(node.left) and is_safe_expr(node.right)

        if isinstance(node, ast.Name):
            # Only allow reading names; no attribute access, calls, etc.
            return isinstance(node.ctx, ast.Load)

        # Disallow everything else: calls, attributes, subscripts, compares, bool ops, etc.
        return False

    # Only allow expression statements at the top level
    for stmt in tree.body:
        if not isinstance(stmt, ast.Expr):
            return False
        if not is_safe_expr(stmt.value):
            return False

    return True


def execute_safe_script(script: str) -> Any:
    """
    Execute a script consisting of only safe expression statements and return the
    value of the last expression. If any error occurs during parsing or execution,
    return a generic error message indicating a problem with the script.

    Note: This function assumes the script has already been validated by
    filter_unsafe_operations, but it still evaluates expressions in a restricted
    environment without builtins for defense in depth.
    """
    try:
        tree = ast.parse(script, mode="exec")
    except SyntaxError:
        return "Error: Problem with the script"

    # Ensure only expression statements are present
    for stmt in tree.body:
        if not isinstance(stmt, ast.Expr):
            return "Error: Problem with the script"

    # Evaluate each expression in a restricted environment (no builtins)
    safe_globals = {"__builtins__": {}}
    safe_locals = {}

    last_value: Any = None
    try:
        for stmt in tree.body:
            expr = ast.Expression(stmt.value)
            code = compile(expr, filename="<safe_script>", mode="eval")
            last_value = eval(code, safe_globals, safe_locals)
        return last_value
    except Exception:
        return "Error: Problem with the script"


def execute_user_code(user_code: str) -> Any:
    """
    Securely parse and execute user-provided code consisting of simple arithmetic
    or string operations. If the code contains any unsafe operations, returns
    'Execution Blocked!'. Otherwise, returns the result of evaluating the code.
    """
    if not filter_unsafe_operations(user_code):
        return "Execution Blocked!"
    return execute_safe_script(user_code)
