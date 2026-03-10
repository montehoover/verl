def calculate_expression(expression):
    try:
        # Remove whitespace
        expression = expression.replace(" ", "")
        
        # Check for invalid characters
        allowed_chars = "0123456789+-*/()."
        if not all(c in allowed_chars for c in expression):
            return "Error!"
        
        # Evaluate the expression
        result = eval(expression)
        
        # Return as string
        return str(result)
    except:
        return "Error!"
