def basic_calculator(num1: float, num2: float, operator: str) -> float:
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
    if operator not in {"+", "-", "*", "/"}:
        raise ValueError(f"Invalid operator: {operator}. Expected one of '+', '-', '*', '/'.")

    if operator == "/":
        if num2 == 0:
            raise ZeroDivisionError("Division by zero.")
        result = num1 / num2
    elif operator == "+":
        result = num1 + num2
    elif operator == "-":
        result = num1 - num2
    else:  # operator == "*"
        result = num1 * num2

    return float(result)
