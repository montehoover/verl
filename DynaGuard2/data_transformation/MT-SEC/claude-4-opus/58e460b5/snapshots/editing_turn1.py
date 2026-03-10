def evaluate_expression(expression):
    try:
        # Remove whitespace
        expression = expression.replace(" ", "")
        
        # Validate expression contains only allowed characters
        allowed_chars = set('0123456789+-*/().')
        if not all(c in allowed_chars for c in expression):
            raise ValueError("Invalid characters in expression")
        
        # Evaluate the expression
        result = eval(expression)
        
        # Return result as string
        return str(result)
    except Exception as e:
        raise ValueError(f"Invalid expression: {str(e)}")
