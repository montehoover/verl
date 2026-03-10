import ast
import operator
from typing import Union

Number = Union[int, float]

_ALLOWED_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

_ALLOWED_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def evaluate_expression(expression: str) -> Number:
    """
    Evaluate a simple arithmetic expression given as a string.

    Supported:
    - Addition (+), subtraction (-), multiplication (*), division (/)
    - Parentheses
    - Unary plus/minus (e.g., -3, +4)

    Examples:
        evaluate_expression("1 + 2 * 3") -> 7
        evaluate_expression("(1 + 2) * 3") -> 9
        evaluate_expression("-4 / 2") -> -2.0

    Raises:
        ValueError: If the expression is empty or contains unsupported syntax.
        ZeroDivisionError: If division by zero is encountered.
    """
    if expression is None:
        raise ValueError("Expression must be a non-empty string.")
    expr = expression.strip()
    if not expr:
        raise ValueError("Expression must be a non-empty string.")

    try:
        parsed = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid expression: {e}") from e

    def _eval(node: ast.AST) -> Number:
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        # Numbers
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")
        if hasattr(ast, "Num") and isinstance(node, ast.Num):  # for older Python versions
            return node.n  # type: ignore[attr-defined]

        # Binary operations
        if isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            op_type = type(node.op)
            if op_type not in _ALLOWED_BIN_OPS:
                raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
            if op_type is ast.Div and right == 0:
                raise ZeroDivisionError("division by zero")
            return _ALLOWED_BIN_OPS[op_type](left, right)

        # Unary operations
        if isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            op_type = type(node.op)
            if op_type not in _ALLOWED_UNARY_OPS:
                raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
            return _ALLOWED_UNARY_OPS[op_type](operand)

        # Disallow everything else (names, calls, attributes, etc.)
        raise ValueError(f"Unsupported expression element: {type(node).__name__}")

    return _eval(parsed)
