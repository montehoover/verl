def calculate(expression):
    try:
        # Remove whitespace
        expression = expression.replace(" ", "")
        
        # Check if expression is empty
        if not expression:
            return "Invalid Expression!"
        
        # Check for valid characters (parentheses already included)
        valid_chars = set("0123456789+-*/.() ")
        if not all(c in valid_chars for c in expression):
            return "Invalid Expression!"
        
        # Check for balanced parentheses
        paren_count = 0
        for char in expression:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            if paren_count < 0:
                return "Invalid Expression!"
        if paren_count != 0:
            return "Invalid Expression!"
        
        # Evaluate the expression
        result = eval(expression)
        
        # Return the result
        return result
        
    except (SyntaxError, ZeroDivisionError, NameError):
        return "Invalid Expression!"
