def basic_calculate(num1: float, num2: float, operator: str) -> float:
    """
    Perform basic arithmetic operations on two numbers.
    
    Args:
        num1: First number
        num2: Second number
        operator: One of '+', '-', '*', '/'
    
    Returns:
        Result of the operation as a float
    """
    if operator == '+':
        return num1 + num2
    elif operator == '-':
        return num1 - num2
    elif operator == '*':
        return num1 * num2
    elif operator == '/':
        if num2 == 0:
            raise ValueError("Division by zero is not allowed")
        return num1 / num2
    else:
        raise ValueError(f"Invalid operator: {operator}. Must be one of '+', '-', '*', '/'")
