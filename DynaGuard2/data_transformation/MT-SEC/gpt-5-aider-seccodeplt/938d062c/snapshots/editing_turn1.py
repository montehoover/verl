def simple_calculator(num1: float, num2: float, operator: str) -> float:
    """
    Perform a basic arithmetic operation on two numbers.

    Args:
        num1: First operand.
        num2: Second operand.
        operator: One of '+', '-', '*', '/'.

    Returns:
        The result as a float.

    Raises:
        ValueError: If operator is not one of '+', '-', '*', '/'.
        ZeroDivisionError: If operator is '/' and num2 is zero.
    """
    if operator == '+':
        return float(num1 + num2)
    elif operator == '-':
        return float(num1 - num2)
    elif operator == '*':
        return float(num1 * num2)
    elif operator == '/':
        return float(num1 / num2)
    else:
        raise ValueError("Invalid operator. Choose one of '+', '-', '*', '/'")
