def calculate_expression(expression):
    try:
        # Only allow basic arithmetic operators and numbers
        allowed_chars = set('0123456789+-*/.()')
        if not all(c in allowed_chars or c.isspace() for c in expression):
            return 'Execution Blocked!'
        
        # Evaluate the expression
        result = eval(expression)
        return result
    except:
        return 'Execution Blocked!'
