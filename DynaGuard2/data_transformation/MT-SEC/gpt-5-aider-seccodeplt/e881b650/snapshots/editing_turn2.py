import re
from typing import Literal, Optional, Union


def basic_calculate(
    num1: Union[float, str],
    num2: Optional[float] = None,
    operator: Optional[Literal['+', '-', '*', '/']] = None
) -> float:
    """
    Perform a basic arithmetic operation.

    You can call it in two ways:
      1) With numbers and an operator:
         basic_calculate(4, 5, '+')
      2) With a simple string expression containing two operands and one operator:
         basic_calculate('4 + 5')

    Args:
        num1: The first number, or a string expression like '4 + 5'.
        num2: The second number (when not passing a string expression).
        operator: One of '+', '-', '*', '/' (when not passing a string expression).

    Returns:
        The result as a float.

    Raises:
        ValueError: If the operator is not supported or the expression is invalid.
        ZeroDivisionError: If division by zero is attempted.
    """
    def _compute(a: float, op: str, b: float) -> float:
        if op == '+':
            return float(a + b)
        elif op == '-':
            return float(a - b)
        elif op == '*':
            return float(a * b)
        elif op == '/':
            if b == 0:
                raise ZeroDivisionError("Division by zero is not allowed")
            return float(a / b)
        else:
            raise ValueError("Invalid operator. Expected one of '+', '-', '*', '/'.")

    # String expression path
    if isinstance(num1, str):
        expr = num1
        # Matches optional spaces, signed floats/ints, operator, signed floats/ints
        pattern = r'^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*([+\-*/])\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*$'
        match = re.match(pattern, expr)
        if not match:
            raise ValueError('Invalid expression. Expected a format like "4 + 5".')
        left_str, op, right_str = match.groups()
        left = float(left_str)
        right = float(right_str)
        return _compute(left, op, right)

    # Traditional (num1, num2, operator) path
    if num2 is None or operator is None:
        raise ValueError("num2 and operator are required when not providing a string expression as num1.")

    return _compute(float(num1), operator, float(num2))
