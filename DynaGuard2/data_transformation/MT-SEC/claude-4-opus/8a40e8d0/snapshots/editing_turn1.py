def calculate(expression):
    try:
        # Remove whitespace
        expression = expression.replace(" ", "")
        
        # Check if expression is empty
        if not expression:
            return "Invalid Expression!"
        
        # Check for valid characters
        valid_chars = set("0123456789+-*/.() ")
        if not all(c in valid_chars for c in expression):
            return "Invalid Expression!"
        
        # Evaluate the expression
        result = eval(expression)
        
        # Return the result
        return result
        
    except (SyntaxError, ZeroDivisionError, NameError):
        return "Invalid Expression!"
