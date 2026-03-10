def simple_calculate(num1: float, num2: float, operator: str) -> float:
    """
    Performs basic arithmetic operations on two numbers.

    Args:
        num1: The first number.
        num2: The second number.
        operator: The operation to perform ('+', '-', '*', '/').

    Returns:
        The result of the arithmetic operation.

    Raises:
        ValueError: If the operator is invalid or if division by zero is attempted.
    """
    if operator == '+':
        return num1 + num2
    elif operator == '-':
        return num1 - num2
    elif operator == '*':
        return num1 * num2
    elif operator == '/':
        if num2 == 0:
            raise ValueError("Cannot divide by zero")
        return num1 / num2
    else:
        raise ValueError(f"Invalid operator: {operator}. Must be one of '+', '-', '*', '/'")
