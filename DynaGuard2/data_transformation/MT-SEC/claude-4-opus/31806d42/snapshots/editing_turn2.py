import re

def basic_calculator(num1, num2=None, operation=None):
    """
    Perform basic arithmetic operations on two numbers or parse and evaluate a string expression.
    
    Args:
        num1 (float or str): First number or string expression like '3 + 4'
        num2 (float, optional): Second number (not used if num1 is a string)
        operation (str, optional): Operation to perform ('+', '-', '*', '/') (not used if num1 is a string)
    
    Returns:
        float: Result of the operation
    """
    # If num1 is a string, parse it as an expression
    if isinstance(num1, str):
        # Regular expression to match number operator number
        pattern = r'^\s*(-?\d+\.?\d*)\s*([\+\-\*/])\s*(-?\d+\.?\d*)\s*$'
        match = re.match(pattern, num1)
        
        if not match:
            raise ValueError(f"Invalid expression format: {num1}")
        
        operand1 = float(match.group(1))
        operator = match.group(2)
        operand2 = float(match.group(3))
        
        return basic_calculator(operand1, operand2, operator)
    
    # Original functionality for separate arguments
    if operation == '+':
        return num1 + num2
    elif operation == '-':
        return num1 - num2
    elif operation == '*':
        return num1 * num2
    elif operation == '/':
        if num2 == 0:
            raise ValueError("Division by zero is not allowed")
        return num1 / num2
    else:
        raise ValueError(f"Invalid operation: {operation}. Use '+', '-', '*', or '/'")
