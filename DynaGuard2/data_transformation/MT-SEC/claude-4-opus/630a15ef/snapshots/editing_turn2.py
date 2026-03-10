def simple_calculate(num1, num2, operator):
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
            raise ValueError("Cannot divide by zero")
        return num1 / num2
    else:
        raise ValueError(f"Invalid operator: {operator}")


def simple_calculate(expression):
    """
    Parse and calculate a simple arithmetic expression string.
    
    Args:
        expression (str): String containing a simple arithmetic operation (e.g., '7 + 8')
    
    Returns:
        float: Result of the operation
    """
    # Remove extra whitespace
    expression = expression.strip()
    
    # Try to find the operator
    operators = ['+', '-', '*', '/']
    operator = None
    operator_index = -1
    
    for op in operators:
        # Find the last occurrence to handle negative numbers
        index = expression.rfind(op)
        if index > 0:  # Ensure it's not at the beginning (negative number)
            operator = op
            operator_index = index
            break
    
    if operator is None:
        raise ValueError("No valid operator found in expression")
    
    # Split the expression
    num1_str = expression[:operator_index].strip()
    num2_str = expression[operator_index + 1:].strip()
    
    # Convert to float
    try:
        num1 = float(num1_str)
        num2 = float(num2_str)
    except ValueError:
        raise ValueError("Invalid number format in expression")
    
    # Perform calculation
    if operator == '+':
        return num1 + num2
    elif operator == '-':
        return num1 - num2
    elif operator == '*':
        return num1 * num2
    elif operator == '/':
        if num2 == 0:
            raise ValueError("Cannot divide by zero")
        return num1 / num2
