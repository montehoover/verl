def basic_calculate(num1: float, num2: float, operator: str) -> float:
    """
    Perform basic arithmetic on two numbers.

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
            raise ZeroDivisionError("division by zero")
        return a / b
    else:
        raise ValueError("Invalid operator. Must be one of '+', '-', '*', '/'.")
