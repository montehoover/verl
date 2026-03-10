def evaluate_expression(expression):
    """
    Evaluate simple arithmetic expressions given as strings.
    
    Args:
        expression: A string containing an arithmetic expression
        
    Returns:
        The result of the expression or 'Error!' if invalid/unsafe
    """
    try:
        # Remove whitespace
        expression = expression.replace(" ", "")
        
        # Check if expression contains only allowed characters
        allowed_chars = "0123456789+-*/()."
        allowed_words = ["True", "False", "and", "or", "not"]
        
        # Create a temporary expression to check characters
        temp_expression = expression
        for word in allowed_words:
            temp_expression = temp_expression.replace(word, "")
        
        if not all(char in allowed_chars for char in temp_expression):
            return 'Error!'
        
        # Check for empty expression
        if not expression:
            return 'Error!'
        
        # Evaluate the expression
        result = eval(expression)
        
        # Handle division by zero
        if isinstance(result, (int, float)) and (result == float('inf') or result == float('-inf')):
            return 'Error!'
        
        return result
        
    except:
        return 'Error!'
