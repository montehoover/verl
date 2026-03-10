def evaluate_math_expression(expression):
    """
    Evaluates a simple arithmetic expression string and returns the result.
    
    Args:
        expression: A string containing a mathematical expression
        
    Returns:
        The calculated result or 'Error!' if the expression is invalid/unsafe
    """
    try:
        # Remove whitespace
        expression = expression.replace(" ", "")
        
        # Check for invalid characters
        allowed_chars = "0123456789+-*/()."
        if not all(c in allowed_chars for c in expression):
            return "Error!"
        
        # Check for empty expression
        if not expression:
            return "Error!"
        
        # Evaluate the expression
        result = eval(expression)
        
        # Return the result
        return result
        
    except:
        return "Error!"
