def evaluate_user_math(math_expr):
    try:
        # Only allow safe characters for basic math and logical operations
        allowed_chars = set('0123456789+-*/.()')
        allowed_words = {'and', 'or', 'not', 'AND', 'OR', 'NOT'}
        
        # Replace logical operators with Python equivalents
        expr = math_expr.replace('AND', 'and').replace('OR', 'or').replace('NOT', 'not')
        
        # Check if expression contains only allowed characters and words
        tokens = expr.replace('(', ' ').replace(')', ' ').split()
        for token in tokens:
            if not (all(c in allowed_chars for c in token) or token in allowed_words):
                return "Processing Failed!"
        
        # Evaluate the expression
        result = eval(expr)
        return result
    except:
        return "Processing Failed!"
