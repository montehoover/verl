def basic_calculate(num1, num2, operator):
    """
    Perform basic arithmetic operations on two numbers.
    
    Args:
        num1 (float): First number
        num2 (float): Second number
        operator (str): Operation to perform ('+', '-', '*', '/')
    
    Returns:
        float: Result of the operation
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
        raise ValueError(f"Invalid operator: {operator}. Use '+', '-', '*', or '/'")
