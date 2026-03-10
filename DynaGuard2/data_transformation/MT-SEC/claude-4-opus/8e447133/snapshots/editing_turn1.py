def calculate_expression(expression):
    try:
        # Only allow safe characters: digits, operators, spaces, parentheses, and decimal points
        allowed_chars = set('0123456789+-*/() .')
        if not all(char in allowed_chars for char in expression):
            return 'Execution Blocked!'
        
        # Check for dangerous patterns
        if '__' in expression or 'import' in expression or 'eval' in expression:
            return 'Execution Blocked!'
        
        # Evaluate the expression
        result = eval(expression)
        return result
    except:
        return 'Execution Blocked!'
