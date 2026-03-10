import re
import ast
import operator as op
from typing import Union, Any

# Allowed binary and unary operations
_ALLOWED_BINOPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}

_ALLOWED_UNARYOPS = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}


def evaluate_user_expression(expression: str) -> Any:
    """
    Safely evaluate a mathematical expression provided by the user.

    Args:
        expression: The string containing the user’s mathematical expression.

    Returns:
        The result of evaluating the mathematical expression.

    Raises:
        ValueError: If the input contains invalid characters, the expression is
                    syntactically incorrect, uses disallowed constructs, or results
                    in an invalid operation (e.g., division by zero).
    """
    if not isinstance(expression, str):
        raise ValueError("Expression must be a string.")

    expr = expression.strip()
    if not expr:
        raise ValueError("Expression is empty.")

    # Quick invalid character check:
    # Allow digits, whitespace, decimal point, arithmetic operators, and parentheses.
    # Also allow 'e'/'E' for scientific notation; AST checks ensure proper usage.
    if re.search(r"[^0-9\.\+\-\*\/\%\(\)\sEe]", expr):
        raise ValueError("Invalid characters in expression.")

    # Parse expression into an AST
    try:
        node = ast.parse(expr, mode='eval')
    except SyntaxError:
        raise ValueError("Incorrect expression.") from None

    def _eval(n: ast.AST) -> Union[int, float]:
        if isinstance(n, ast.Expression):
            return _eval(n.body)

        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float)):
                return n.value
            raise ValueError("Invalid constant in expression.")

        if isinstance(n, ast.BinOp):
            left = _eval(n.left)
            right = _eval(n.right)
            op_type = type(n.op)
            if op_type not in _ALLOWED_BINOPS:
                raise ValueError("Operator not allowed.")
            try:
                return _ALLOWED_BINOPS[op_type](left, right)
            except ZeroDivisionError:
                raise ValueError("Division by zero.") from None

        if isinstance(n, ast.UnaryOp):
            operand = _eval(n.operand)
            op_type = type(n.op)
            if op_type not in _ALLOWED_UNARYOPS:
                raise ValueError("Operator not allowed.")
            return _ALLOWED_UNARYOPS[op_type](operand)

        # Disallow any other constructs: names, calls, attributes, tuples, etc.
        raise ValueError("Invalid expression.")

    return _eval(node)
