def evaluate_expression(expression):
    """
    Evaluates simple arithmetic expressions with addition and subtraction.
    Returns the result as a string, or 'Invalid Expression!' if invalid.
    """
    try:
        # Remove spaces
        expression = expression.replace(" ", "")
        
        # Check if expression is empty
        if not expression:
            return "Invalid Expression!"
        
        # Check for invalid characters
        valid_chars = "0123456789+-"
        for char in expression:
            if char not in valid_chars:
                return "Invalid Expression!"
        
        # Check for invalid patterns
        if expression.startswith('+') or expression.startswith('-'):
            return "Invalid Expression!"
        if expression.endswith('+') or expression.endswith('-'):
            return "Invalid Expression!"
        
        # Check for consecutive operators
        for i in range(len(expression) - 1):
            if expression[i] in "+-" and expression[i+1] in "+-":
                return "Invalid Expression!"
        
        # Evaluate the expression
        result = eval(expression)
        return str(result)
    except:
        return "Invalid Expression!"
