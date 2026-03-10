import ast
import operator
from numbers import Real

_ALLOWED_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def execute_operation(operation: str):
    """
    Execute a simple arithmetic operation from a string and return the result.

    Supported:
      - Numbers (int, float)
      - Binary ops: +, -, *, /, //, %, **
      - Unary ops: +, -
      - Parentheses

    Parameters:
        operation (str): Expression like "2 + 3" or "(1 + 2) * 3"

    Returns:
        int or float: The computed result, following Python arithmetic semantics.

    Raises:
        TypeError: If operation is not a string.
        ValueError: If the expression contains unsupported syntax.
        ZeroDivisionError: If a division by zero occurs.
    """
    if not isinstance(operation, str):
        raise TypeError("operation must be a string")

    try:
        tree = ast.parse(operation, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid expression: {e}") from e

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        # Numeric literal
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Only int and float constants are allowed")

        # For Python versions where numbers may appear as ast.Num
        if hasattr(ast, "Num") and isinstance(node, ast.Num):  # type: ignore[attr-defined]
            return node.n  # type: ignore[no-any-return]

        # Unary operations (+x, -x)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY_OPS:
            operand = _eval(node.operand)
            if not isinstance(operand, Real):
                raise ValueError("Unary operations allowed only on numbers")
            return _ALLOWED_UNARY_OPS[type(node.op)](operand)

        # Binary operations (x + y, x * y, etc.)
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BIN_OPS:
            left = _eval(node.left)
            right = _eval(node.right)
            if not isinstance(left, Real) or not isinstance(right, Real):
                raise ValueError("Binary operations allowed only on numbers")
            func = _ALLOWED_BIN_OPS[type(node.op)]
            return func(left, right)

        # Disallow all other constructs (names, calls, attributes, etc.)
        raise ValueError(f"Unsupported expression element: {type(node).__name__}")

    return _eval(tree)
