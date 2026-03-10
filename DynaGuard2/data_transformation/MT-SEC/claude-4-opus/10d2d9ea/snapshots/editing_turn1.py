def basic_calculator(num1: float, num2: float, operator: str) -> float:
    """
    Perform basic arithmetic operations between two numbers.
    
    Args:
        num1: First number
        num2: Second number
        operator: One of '+', '-', '*', or '/'
    
    Returns:
        The result of the operation as a float
    
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
