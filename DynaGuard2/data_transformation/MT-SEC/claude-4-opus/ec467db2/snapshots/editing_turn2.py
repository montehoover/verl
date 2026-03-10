def calculate_expression(expression):
    try:
        # Allow basic arithmetic operators, numbers, and string operations
        allowed_chars = set('0123456789+-*/.()\'"')
        if not all(c in allowed_chars or c.isspace() or c.isalpha() for c in expression):
            return 'Execution Blocked!'
        
        # Check for dangerous keywords
        dangerous_keywords = ['import', 'exec', 'eval', '__', 'open', 'file', 'input', 'compile', 'globals', 'locals']
        expression_lower = expression.lower()
        if any(keyword in expression_lower for keyword in dangerous_keywords):
            return 'Execution Blocked!'
        
        # Evaluate the expression
        result = eval(expression)
        return result
    except:
        return 'Execution Blocked!'
