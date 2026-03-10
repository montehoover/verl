def evaluate_expression(expression):
    try:
        # Remove whitespace
        expression = expression.replace(" ", "")
        
        # Check if expression is empty
        if not expression:
            return "Invalid Expression"
        
        # Check for invalid characters
        valid_chars = set("0123456789+-.")
        if not all(c in valid_chars for c in expression):
            return "Invalid Expression"
        
        # Check for consecutive operators
        for i in range(len(expression) - 1):
            if expression[i] in "+-" and expression[i + 1] in "+-":
                if not (i == 0 or expression[i - 1] in "+-"):
                    return "Invalid Expression"
        
        # Evaluate the expression
        result = eval(expression)
        return result
    except:
        return "Invalid Expression"
