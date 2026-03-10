import re
from typing import Optional, Union

def basic_calculate(num1: Union[float, str], num2: Optional[float] = None, operator: Optional[str] = None) -> float:
    """
    Perform basic arithmetic on two numbers or parse a simple expression string.

    Usage:
        basic_calculate(5, 3, '+') -> 8.0
        basic_calculate("5 + 3") -> 8.0

    Args:
        num1: First operand as float, or a string expression like '5 + 3'.
        num2: Second operand (optional when using string form).
        operator: One of '+', '-', '*', '/' (optional when using string form).

    Returns:
        The result as a float.

    Raises:
        ValueError: If operator or expression is invalid.
        ZeroDivisionError: If division by zero is attempted.
    """
    def compute(a: float, b: float, op: str) -> float:
        if op == '+':
            return a + b
        elif op == '-':
            return a - b
        elif op == '*':
            return a * b
        elif op == '/':
            if b == 0.0:
                raise ZeroDivisionError("division by zero")
            return a / b
        else:
            raise ValueError("Invalid operator. Must be one of '+', '-', '*', '/'.")

    # String expression mode: e.g., "5 + 3"
    if isinstance(num1, str) and num2 is None and operator is None:
        expr = num1.strip()
        match = re.match(r'^\s*([+-]?\d+(?:\.\d+)?)\s*([+\-*/])\s*([+-]?\d+(?:\.\d+)?)\s*$', expr)
        if not match:
            raise ValueError("Invalid expression format. Expected '<number> <operator> <number>'.")
        left_str, op, right_str = match.groups()
        a = float(left_str)
        b = float(right_str)
        return compute(a, b, op)

    # Traditional mode: num1, num2, operator
    if num2 is None or operator is None:
        raise ValueError("Missing arguments. Provide either a string expression or num1, num2, and operator.")

    a = float(num1)
    b = float(num2)
    return compute(a, b, operator)
