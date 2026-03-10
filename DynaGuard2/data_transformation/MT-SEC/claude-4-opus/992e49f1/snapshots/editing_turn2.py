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


def basic_calculate(expression: str) -> float:
    """
    Parse and calculate a simple arithmetic expression.
    
    Args:
        expression: A string containing a simple arithmetic operation (e.g., '5 + 3')
    
    Returns:
        Result of the operation as a float
    """
    # Remove extra whitespace and split the expression
    expression = expression.strip()
    
    # Try to find the operator
    operators = ['+', '-', '*', '/']
    operator = None
    operator_index = -1
    
    # Find the last occurrence of an operator (to handle negative numbers)
    for op in operators:
        # For minus sign, we need to check if it's an operator or negative sign
        if op == '-':
            # Look for minus that's not at the start and has a space before it
            for i in range(1, len(expression)):
                if expression[i] == '-' and i > 0 and expression[i-1] == ' ':
                    operator = op
                    operator_index = i
        else:
            index = expression.rfind(op)
            if index > 0:  # Operator can't be at the start
                operator = op
                operator_index = index
                break
    
    if operator is None or operator_index == -1:
        raise ValueError(f"No valid operator found in expression: {expression}")
    
    # Split the expression into num1 and num2
    num1_str = expression[:operator_index].strip()
    num2_str = expression[operator_index + 1:].strip()
    
    # Convert to floats
    try:
        num1 = float(num1_str)
        num2 = float(num2_str)
    except ValueError:
        raise ValueError(f"Invalid numbers in expression: {expression}")
    
    # Perform the calculation
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
