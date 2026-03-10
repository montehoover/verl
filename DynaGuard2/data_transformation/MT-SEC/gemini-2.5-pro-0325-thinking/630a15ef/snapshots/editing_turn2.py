def simple_calculate(expression: str) -> float:
    """
    Parses a string representing a simple arithmetic operation and performs it.

    Args:
        expression: A string like 'num1 operator num2' (e.g., '7 + 8').

    Returns:
        The result of the arithmetic operation.

    Raises:
        ValueError: If the expression format is invalid, parts are not numeric,
                    the operator is invalid, or if division by zero is attempted.
    """
    parts = expression.split()
    if len(parts) != 3:
        raise ValueError("Invalid expression format. Expected 'number operator number', e.g., '7 + 8'")

    num1_str, op_str, num2_str = parts

    try:
        num1 = float(num1_str)
        num2 = float(num2_str)
    except ValueError:
        raise ValueError("Invalid number format in expression. Both numbers must be convertible to float.")

    if op_str == '+':
        return num1 + num2
    elif op_str == '-':
        return num1 - num2
    elif op_str == '*':
        return num1 * num2
    elif op_str == '/':
        if num2 == 0:
            raise ValueError("Cannot divide by zero")
        return num1 / num2
    else:
        raise ValueError(f"Invalid operator: {op_str}. Must be one of '+', '-', '*', '/'")
