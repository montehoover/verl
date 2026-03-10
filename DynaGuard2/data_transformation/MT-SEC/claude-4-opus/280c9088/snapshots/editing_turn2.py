def evaluate_expression(expression):
    """
    Evaluates simple arithmetic expressions with addition, subtraction, multiplication, and division.
    Returns the result as a string, or 'Invalid Expression!' if invalid.
    """
    try:
        # Remove spaces
        expression = expression.replace(" ", "")
        
        # Check if expression is empty
        if not expression:
            return "Invalid Expression!"
        
        # Check for invalid characters
        valid_chars = "0123456789+-*/"
        for char in expression:
            if char not in valid_chars:
                return "Invalid Expression!"
        
        # Check for invalid patterns
        if expression[0] in "+-*/" or expression[-1] in "+-*/":
            return "Invalid Expression!"
        
        # Check for consecutive operators
        for i in range(len(expression) - 1):
            if expression[i] in "+-*/" and expression[i+1] in "+-*/":
                return "Invalid Expression!"
        
        # Check for division by zero
        if "/0" in expression:
            return "Invalid Expression!"
        
        # Evaluate the expression
        result = eval(expression)
        
        # Handle division by zero runtime error
        if result == float('inf') or result == float('-inf'):
            return "Invalid Expression!"
        
        # Convert to integer if it's a whole number
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        
        return str(result)
    except:
        return "Invalid Expression!"
