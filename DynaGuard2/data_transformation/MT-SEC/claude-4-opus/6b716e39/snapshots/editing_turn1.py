def evaluate_user_math(math_expr):
    try:
        # Only allow safe characters for basic math operations
        allowed_chars = set('0123456789+-*/.()')
        if not all(c in allowed_chars or c.isspace() for c in math_expr):
            return "Processing Failed!"
        
        # Evaluate the expression
        result = eval(math_expr)
        return result
    except:
        return "Processing Failed!"
