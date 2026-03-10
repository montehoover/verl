def evaluate_expression(expression):
    try:
        # Check if expression is empty
        if not expression.strip():
            return "Invalid Expression"
        
        # Check if it's a string concatenation
        if "'" in expression or '"' in expression:
            # Validate string expression
            temp_expr = expression.strip()
            
            # Basic validation for string expressions
            if temp_expr.count("'") % 2 != 0 and temp_expr.count('"') % 2 != 0:
                return "Invalid Expression"
            
            # Evaluate string expression
            result = eval(expression)
            return result
        else:
            # Handle arithmetic expression
            # Remove whitespace
            expression = expression.replace(" ", "")
            
            # Check if expression is empty after removing spaces
            if not expression:
                return "Invalid Expression"
            
            # Check for invalid characters in arithmetic expression
            valid_chars = set("0123456789+-.")
            if not all(c in valid_chars for c in expression):
                return "Invalid Expression"
            
            # Check for consecutive operators
            for i in range(len(expression) - 1):
                if expression[i] in "+-" and expression[i + 1] in "+-":
                    if not (i == 0 or expression[i - 1] in "+-"):
                        return "Invalid Expression"
            
            # Evaluate the arithmetic expression
            result = eval(expression)
            return result
    except:
        return "Invalid Expression"
