def basic_arithmetic(num1, num2=None, operator=None):
    """
    Perform basic arithmetic operations on two numbers.
    
    Args:
        num1: Either a float for the first number, or a string representing the entire operation (e.g., '5 + 7')
        num2 (float, optional): Second number (not needed if num1 is a string)
        operator (str, optional): Operation to perform ('+', '-', '*', '/') (not needed if num1 is a string)
    
    Returns:
        float: Result of the arithmetic operation
    """
    # If num1 is a string, parse it
    if isinstance(num1, str):
        # Split the string and extract operands and operator
        parts = num1.strip().split()
        if len(parts) != 3:
            raise ValueError("String must be in format 'number operator number' (e.g., '5 + 7')")
        
        try:
            parsed_num1 = float(parts[0])
            parsed_operator = parts[1]
            parsed_num2 = float(parts[2])
        except ValueError:
            raise ValueError("Invalid number format in string")
        
        # Recursively call the function with parsed values
        return basic_arithmetic(parsed_num1, parsed_num2, parsed_operator)
    
    # Original functionality for separate arguments
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
