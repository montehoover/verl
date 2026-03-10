from typing import Union

def simple_calculate(num1: Union[int, float], num2: Union[int, float], operator: str) -> float:
    """
    Perform a basic arithmetic operation on two numbers.

    Args:
        num1: First number.
        num2: Second number.
        operator: One of '+', '-', '*', '/'.

    Returns:
        The result as a float.

    Raises:
        ValueError: If the operator is not supported.
        ZeroDivisionError: If division by zero is attempted.
    """
    a = float(num1)
    b = float(num2)

    if operator == '+':
        return a + b
    elif operator == '-':
        return a - b
    elif operator == '*':
        return a * b
    elif operator == '/':
        if b == 0.0:
            raise ZeroDivisionError("Division by zero.")
        return a / b
    else:
        raise ValueError(f"Unsupported operator: {operator}. Use one of '+', '-', '*', '/'.")
