def calculate_expression(expression):
    """
    Calculate the result of a simple arithmetic expression.
    
    Args:
        expression: A string containing a simple arithmetic expression
        
    Returns:
        The result of the expression, or 'Execution Blocked!' if invalid/unsafe
    """
    try:
        # Remove whitespace
        expression = expression.strip()
        
        # Check if expression is empty
        if not expression:
            return 'Execution Blocked!'
        
        # Only allow digits, operators, parentheses, decimal points, and spaces
        allowed_chars = set('0123456789+-*/()., ')
        if not all(c in allowed_chars for c in expression):
            return 'Execution Blocked!'
        
        # Block dangerous functions and keywords
        dangerous_patterns = ['__', 'import', 'eval', 'exec', 'open', 'file', 
                            'input', 'raw_input', 'compile', 'globals', 'locals']
        expression_lower = expression.lower()
        if any(pattern in expression_lower for pattern in dangerous_patterns):
            return 'Execution Blocked!'
        
        # Evaluate the expression
        result = eval(expression)
        return result
        
    except:
        return 'Execution Blocked!'
