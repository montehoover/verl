from typing import Literal


def basic_calculate(num1: float, num2: float, operator: Literal['+', '-', '*', '/']) -> float:
    """
    Perform a basic arithmetic operation on two numbers.

    Args:
        num1: The first number.
        num2: The second number.
        operator: One of '+', '-', '*', '/'.

    Returns:
        The result as a float.

    Raises:
        ValueError: If the operator is not supported.
        ZeroDivisionError: If division by zero is attempted.
    """
    if operator == '+':
        return float(num1 + num2)
    elif operator == '-':
        return float(num1 - num2)
    elif operator == '*':
        return float(num1 * num2)
    elif operator == '/':
        if num2 == 0:
            raise ZeroDivisionError("Division by zero is not allowed")
        return float(num1 / num2)
    else:
        raise ValueError("Invalid operator. Expected one of '+', '-', '*', '/'.")
