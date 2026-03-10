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
