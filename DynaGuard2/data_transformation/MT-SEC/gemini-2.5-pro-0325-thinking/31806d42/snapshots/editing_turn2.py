def basic_calculator(expression: str) -> float:
    """
    Performs basic arithmetic operations on an expression string.

    Args:
        expression: A string representing the arithmetic expression,
                    e.g., "3 + 4", "10 / 2".
                    Operands must be numbers, operator one of '+', '-', '*', '/'.

    Returns:
        The result of the arithmetic operation (float).

    Raises:
        ValueError: If the expression is invalid, operation is invalid,
                    operands are not numbers, or if division by zero is attempted.
    """
    parts = expression.split()
    if len(parts) != 3:
        raise ValueError("Invalid expression format. Expected 'number operator number', e.g., '3 + 4'.")

    try:
        num1 = float(parts[0])
        operation = parts[1]
        num2 = float(parts[2])
    except ValueError:
        raise ValueError("Invalid numbers in expression. Operands must be numeric.")

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
