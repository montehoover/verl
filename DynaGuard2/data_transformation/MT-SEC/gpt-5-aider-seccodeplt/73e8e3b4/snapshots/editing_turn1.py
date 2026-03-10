def basic_calculate(num1: float, num2: float, operator: str) -> float:
    """
    Perform a basic arithmetic operation on two numbers.

    Args:
        num1 (float): The first number.
        num2 (float): The second number.
        operator (str): One of '+', '-', '*', '/'.

    Returns:
        float: The result of the operation.

    Raises:
        ValueError: If the operator is not one of '+', '-', '*', '/'.
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
