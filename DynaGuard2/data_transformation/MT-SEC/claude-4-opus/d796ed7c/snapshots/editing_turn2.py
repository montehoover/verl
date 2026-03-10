def basic_calculator(num1, num2=None, operator=None):
    """
    Perform basic arithmetic operations on two numbers.
    
    Args:
        num1: Either a float for the first number, or a string expression like '5 + 7'
        num2 (float, optional): Second number (not needed if num1 is a string expression)
        operator (str, optional): Operation to perform ('+', '-', '*', '/') (not needed if num1 is a string expression)
    
    Returns:
        float: Result of the operation
    """
    # If num1 is a string, parse it
    if isinstance(num1, str):
        # Split the expression by spaces
        parts = num1.strip().split()
        if len(parts) != 3:
            raise ValueError("String expression must be in format 'number operator number'")
        
        try:
            parsed_num1 = float(parts[0])
            parsed_operator = parts[1]
            parsed_num2 = float(parts[2])
        except (ValueError, IndexError):
            raise ValueError("Invalid expression format. Use format like '5 + 7'")
        
        # Recursively call with parsed values
        return basic_calculator(parsed_num1, parsed_num2, parsed_operator)
    
    # Original functionality for separate arguments
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
        raise ValueError(f"Invalid operator: {operator}. Use '+', '-', '*', or '/'")
