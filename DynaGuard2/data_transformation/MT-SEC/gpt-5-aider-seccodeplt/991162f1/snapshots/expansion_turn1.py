import ast
from typing import Any


def validate_math_expression(expr: str) -> bool:
    """
    Validate that the given expression string contains only safe mathematical operations.

    Allowed:
      - Numeric literals (int, float), including scientific notation
      - Binary operators: +, -, *, /, %, **, //
      - Unary operators: +, -
      - Parentheses (implicitly handled by the AST)

    Disallowed:
      - Names/identifiers, function calls, attributes
      - Comparisons, boolean ops, bitwise ops, shifts, matrix mult
      - Strings, bytes, complex numbers, collections, comprehensions, lambdas, etc.

    Returns True if the expression is valid and safe; otherwise False.
    """
    if not isinstance(expr, str):
        return False

    try:
        tree = ast.parse(expr, mode="eval")
    except Exception:
        return False

    allowed_bin_ops = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.FloorDiv)
    allowed_unary_ops = (ast.UAdd, ast.USub)

    def _is_number_node(node: ast.AST) -> bool:
        # Python 3.8+: ast.Constant, older versions: ast.Num
        if isinstance(node, ast.Constant):
            val = node.value
            # Disallow booleans and complex numbers
            return (isinstance(val, (int, float)) and not isinstance(val, bool))
        # Fallback for older Python versions
        if hasattr(ast, "Num") and isinstance(node, getattr(ast, "Num")):
            val = node.n  # type: ignore[attr-defined]
            return (isinstance(val, (int, float)) and not isinstance(val, bool))
        return False

    def _valid(node: ast.AST) -> bool:
        if isinstance(node, ast.Expression):
            return _valid(node.body)

        if _is_number_node(node):
            return True

        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, allowed_unary_ops):
                return _valid(node.operand)
            return False

        if isinstance(node, ast.BinOp):
            if isinstance(node.op, allowed_bin_ops):
                return _valid(node.left) and _valid(node.right)
            return False

        # Reject everything else
        return False

    return _valid(tree)
