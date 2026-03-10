import re
from typing import Optional, Tuple, Union

def simple_calculate(num1: Union[int, float, str], num2: Optional[Union[int, float]] = None, operator: Optional[str] = None) -> float:
    """
    Perform a basic arithmetic operation on two numbers.

    Usage:
        - simple_calculate(num1, num2, operator)
        - simple_calculate("7 + 8")

    Args:
        num1: First number, or an expression string like "7 + 8".
        num2: Second number (ignored if num1 is a string expression).
        operator: One of '+', '-', '*', '/' (ignored if num1 is a string expression).

    Returns:
        The result as a float.

    Raises:
        ValueError: If the operator or expression is not supported/valid.
        ZeroDivisionError: If division by zero is attempted.
    """

    def _parse_expression(expr: str) -> Tuple[float, float, str]:
        pattern = r'^\s*([+-]?\d+(?:\.\d+)?)\s*([+\-*/])\s*([+-]?\d+(?:\.\d+)?)\s*$'
        match = re.match(pattern, expr)
        if not match:
            raise ValueError('Invalid expression format. Expected like "7 + 8".')
        left_str, op, right_str = match.groups()
        return float(left_str), float(right_str), op

    if isinstance(num1, str) and num2 is None and operator is None:
        a, b, op = _parse_expression(num1)
    else:
        a = float(num1)
        if num2 is None or operator is None:
            raise ValueError("When providing numeric arguments, both num2 and operator are required.")
        b = float(num2)
        op = operator

    if op == '+':
        return a + b
    elif op == '-':
        return a - b
    elif op == '*':
        return a * b
    elif op == '/':
        if b == 0.0:
            raise ZeroDivisionError("Division by zero.")
        return a / b
    else:
        raise ValueError(f"Unsupported operator: {op}. Use one of '+', '-', '*', '/'.")
