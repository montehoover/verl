import ast
import operator as _op
from typing import Union


__all__ = ["execute_operation"]

# Allowed binary and unary operators for safe evaluation
_BIN_OPS = {
    ast.Add: _op.add,
    ast.Sub: _op.sub,
    ast.Mult: _op.mul,
    ast.Div: _op.truediv,
    ast.FloorDiv: _op.floordiv,
    ast.Mod: _op.mod,
    ast.Pow: _op.pow,
}

_UNARY_OPS = {
    ast.UAdd: _op.pos,
    ast.USub: _op.neg,
}


def _eval_ast(node: ast.AST) -> Union[int, float]:
    """
    Recursively and safely evaluate an arithmetic AST node.
    Supports numbers, parentheses, +, -, *, /, //, %, **, and unary + / -.
    """
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BIN_OPS:
            raise ValueError(f"Operator '{op_type.__name__}' is not allowed.")
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        return _BIN_OPS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise ValueError(f"Unary operator '{op_type.__name__}' is not allowed.")
        operand = _eval_ast(node.operand)
        return _UNARY_OPS[op_type](operand)

    if isinstance(node, ast.Constant):
        value = node.value
        # Disallow booleans and complex numbers; allow ints and floats only.
        if isinstance(value, bool) or isinstance(value, complex):
            raise ValueError("Only real numeric literals are allowed.")
        if isinstance(value, (int, float)):
            return value
        raise ValueError("Only numeric literals are allowed.")

    # For Python versions where numbers might be represented as ast.Num
    if hasattr(ast, "Num") and isinstance(node, getattr(ast, "Num")):
        value = node.n
        if isinstance(value, bool) or isinstance(value, complex):
            raise ValueError("Only real numeric literals are allowed.")
        if isinstance(value, (int, float)):
            return value
        raise ValueError("Only numeric literals are allowed.")

    # Explicitly forbid names, calls, attributes, comprehensions, etc.
    raise ValueError(f"Disallowed expression: {ast.dump(node, include_attributes=False)}")


def execute_operation(operation: str) -> float:
    """
    Evaluate an arithmetic expression represented as a string, e.g., "3 * (4 + 5)".

    Supported:
      - Numbers (integers and floats, including scientific notation)
      - Parentheses
      - Operators: +, -, *, /, //, %, ** (exponent)
      - Unary + and -

    Args:
        operation: The expression string, e.g., "5 + 3", "3*(4+5)", "-1.2e3 / 2".

    Returns:
        The result as a float.

    Raises:
        TypeError: If operation is not a string.
        ValueError: If parsing fails or the expression contains disallowed constructs.
        ZeroDivisionError: If division by zero occurs.
    """
    if not isinstance(operation, str):
        raise TypeError("operation must be a string")

    try:
        parsed = ast.parse(operation, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Invalid arithmetic expression.") from exc

    result = _eval_ast(parsed)
    return float(result)
