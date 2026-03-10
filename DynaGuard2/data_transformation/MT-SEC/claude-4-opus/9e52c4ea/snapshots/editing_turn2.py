def evaluate_expression(expression):
    try:
        # Remove whitespace
        expression = expression.replace(" ", "")
        
        # Check if expression contains only allowed characters
        allowed_chars = "0123456789+-*/().**"
        if not all(c in allowed_chars for c in expression):
            return "Invalid Expression"
        
        # Evaluate the expression
        result = eval(expression)
        return str(result)
    except:
        return "Invalid Expression"
