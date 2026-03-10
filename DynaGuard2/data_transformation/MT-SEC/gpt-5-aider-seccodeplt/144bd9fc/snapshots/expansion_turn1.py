import ast
from typing import Union


def parse_math_expression(expression: str) -> bool:
    """
    Validate that the given expression is a safe, basic arithmetic expression.

    Rules:
    - Only digits, decimal points, whitespace, parentheses, and operators + - * / are allowed.
    - No variables, function calls, attribute access, or other Python constructs.
    - Allows unary + and - operators.
    - Disallows exponentiation (**) and floor division (//).
    - Expression must not be empty or only whitespace.

    Returns:
        True if the expression is valid and safe, False otherwise.
    """
    if not isinstance(expression, str):
        return False

    expr = expression.strip()
    if not expr:
        return False

    # Character whitelist to ensure only numbers and basic arithmetic symbols are present.
    allowed_chars = set("0123456789.+-*/() \t\r\n")
    if not set(expr).issubset(allowed_chars):
        return False

    # Quick early rejects of multi-char operators we don't allow.
    if "**" in expr or "//" in expr:
        return False

    # Must contain at least one digit to avoid inputs like "+" or "()"
    if not any(ch.isdigit() for ch in expr):
        return False

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return False

    allowed_bin_ops = (ast.Add, ast.Sub, ast.Mult, ast.Div)
    allowed_unary_ops = (ast.UAdd, ast.USub)

    def is_safe(node: ast.AST) -> bool:
        if isinstance(node, ast.Expression):
            return is_safe(node.body)

        if isinstance(node, ast.BinOp):
            return (
                isinstance(node.op, allowed_bin_ops)
                and is_safe(node.left)
                and is_safe(node.right)
            )

        if isinstance(node, ast.UnaryOp):
            return isinstance(node.op, allowed_unary_ops) and is_safe(node.operand)

        if isinstance(node, ast.Constant):
            # Only allow int/float constants. Note: bool is subclass of int; exclude it.
            return isinstance(node.value, (int, float)) and not isinstance(node.value, bool)

        # Parentheses don't appear explicitly in the AST; they affect grouping only.

        # Disallow all other nodes: Names, Calls, Attributes, Subscripts, etc.
        return False

    return is_safe(tree)
