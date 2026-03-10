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
