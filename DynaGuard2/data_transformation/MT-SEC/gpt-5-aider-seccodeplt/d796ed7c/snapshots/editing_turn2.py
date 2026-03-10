import re
from typing import Optional, Union

def basic_calculator(num1: Union[float, str], num2: Optional[float] = None, operator: Optional[str] = None) -> float:
    """
    Perform a basic arithmetic operation.

    Usage:
        - basic_calculator(5.0, 7.0, '+')
        - basic_calculator("5 + 7")

    Args:
        num1: First operand as float, or an expression string like '5 + 7'.
        num2: Second operand (optional if using expression string).
        operator: One of '+', '-', '*', '/' (optional if using expression string).

    Returns:
        The result as a float.

    Raises:
        ValueError: If operator or expression is invalid.
        ZeroDivisionError: If division by zero is attempted.
        TypeError: If arguments do not match expected formats.
    """
    # Expression mode
    if isinstance(num1, str) and num2 is None and operator is None:
        expr = num1.strip()
        pattern = r'^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*([+\-*/])\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*$'
        match = re.match(pattern, expr)
        if not match:
            raise ValueError("Invalid expression format. Expected a simple binary expression like '5 + 7'.")
        a_str, op, b_str = match.groups()
        a = float(a_str)
        b = float(b_str)
        return basic_calculator(a, b, op)

    # Direct args mode
    if num2 is None or operator is None:
        raise TypeError("Expected (num1: float, num2: float, operator: str) or a single expression string like '5 + 7'.")

    if operator not in {"+", "-", "*", "/"}:
        raise ValueError(f"Invalid operator: {operator}. Expected one of '+', '-', '*', '/'.")

    a = float(num1)
    b = float(num2)

    if operator == "/":
        if b == 0:
            raise ZeroDivisionError("Division by zero.")
        result = a / b
    elif operator == "+":
        result = a + b
    elif operator == "-":
        result = a - b
    else:  # operator == "*"
        result = a * b

    return float(result)
