def basic_calculate(num1: float, num2: float, operator: str) -> float:
    """
    Perform basic arithmetic operations on two numbers.
    
    Args:
        num1 (float): First number
        num2 (float): Second number
        operator (str): Operation to perform ('+', '-', '*', '/')
    
    Returns:
        float: Result of the operation
    
    Raises:
        ValueError: If operator is not valid
        ZeroDivisionError: If dividing by zero
    """
    if operator == '+':
        return num1 + num2
    elif operator == '-':
        return num1 - num2
    elif operator == '*':
        return num1 * num2
    elif operator == '/':
        if num2 == 0:
            raise ZeroDivisionError("Cannot divide by zero")
        return num1 / num2
    else:
        raise ValueError(f"Invalid operator: {operator}. Must be one of '+', '-', '*', '/'")


def calculate_expression(expression: str) -> float:
    """
    Calculate a simple mathematical expression from a string.
    
    Args:
        expression (str): String expression like '4 + 5'
    
    Returns:
        float: Result of the calculation
    
    Raises:
        ValueError: If expression format is invalid
    """
    # Remove extra spaces and split the expression
    parts = expression.strip().split()
    
    if len(parts) != 3:
        raise ValueError("Expression must be in format 'number operator number'")
    
    try:
        num1 = float(parts[0])
        operator = parts[1]
        num2 = float(parts[2])
    except (ValueError, IndexError):
        raise ValueError("Invalid expression format")
    
    return basic_calculate(num1, num2, operator)
