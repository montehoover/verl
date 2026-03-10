from typing import Literal

Operator = Literal['+', '-', '*', '/']


def basic_arithmetic(num1: float, num2: float, operator: Operator) -> float:
    """
    Perform a basic arithmetic operation on two numbers.

    Args:
        num1: First operand.
        num2: Second operand.
        operator: One of '+', '-', '*', '/'.

    Returns:
        The result as a float.

    Raises:
        ValueError: If operator is invalid.
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
            raise ZeroDivisionError("Division by zero is not allowed.")
        return float(num1 / num2)
    else:
        raise ValueError(f"Unsupported operator: {operator!r}. Expected one of '+', '-', '*', '/'.")
