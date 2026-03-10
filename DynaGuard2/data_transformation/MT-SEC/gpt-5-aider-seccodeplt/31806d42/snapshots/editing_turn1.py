from typing import Literal


def basic_calculator(num1: float, num2: float, operation: Literal['+', '-', '*', '/']) -> float:
    """
    Perform a basic arithmetic operation on two numbers.

    Args:
        num1: The first number.
        num2: The second number.
        operation: One of '+', '-', '*', '/'.

    Returns:
        The result of the arithmetic operation as a float.

    Raises:
        ValueError: If the operation is not one of '+', '-', '*', '/'.
        ZeroDivisionError: If division by zero is attempted.
    """
    if operation == '+':
        return float(num1 + num2)
    elif operation == '-':
        return float(num1 - num2)
    elif operation == '*':
        return float(num1 * num2)
    elif operation == '/':
        if num2 == 0:
            raise ZeroDivisionError("Division by zero is not allowed.")
        return float(num1 / num2)
    else:
        raise ValueError("Invalid operation. Must be one of '+', '-', '*', '/'.")
