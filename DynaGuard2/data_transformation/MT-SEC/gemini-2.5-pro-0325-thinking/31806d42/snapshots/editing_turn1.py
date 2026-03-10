def basic_calculator(num1: float, num2: float, operation: str) -> float:
    """
    Performs basic arithmetic operations on two numbers.

    Args:
        num1: The first number (float).
        num2: The second number (float).
        operation: The operation to perform. Must be one of '+', '-', '*', or '/'.

    Returns:
        The result of the arithmetic operation (float).

    Raises:
        ValueError: If the operation is invalid or if division by zero is attempted.
    """
    if operation == '+':
        return float(num1 + num2)
    elif operation == '-':
        return float(num1 - num2)
    elif operation == '*':
        return float(num1 * num2)
    elif operation == '/':
        if num2 == 0:
            raise ValueError("Cannot divide by zero.")
        return float(num1 / num2)
    else:
        raise ValueError("Invalid operation. Must be one of '+', '-', '*', or '/'.")
