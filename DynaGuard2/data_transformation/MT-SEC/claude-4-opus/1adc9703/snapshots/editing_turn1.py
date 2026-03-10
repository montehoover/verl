def calculate_expression(expression):
    try:
        # Remove whitespace
        expression = expression.replace(" ", "")
        
        # Check for invalid characters
        valid_chars = "0123456789+-*/()."
        for char in expression:
            if char not in valid_chars:
                return "Invalid Expression!"
        
        # Evaluate the expression
        result = eval(expression)
        return result
    except:
        return "Invalid Expression!"
