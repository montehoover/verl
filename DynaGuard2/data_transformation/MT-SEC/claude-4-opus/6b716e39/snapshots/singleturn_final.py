def evaluate_user_math(math_expr: str) -> str:
    """
    Safely evaluates a mathematical expression provided as a string.
    
    Args:
        math_expr: A string representing the user-supplied expression that requires evaluation.
        
    Returns:
        The outcome of the evaluated expression in string format, or 'Processing Failed!' 
        if there is any error or a security issue is detected.
    """
    try:
        # Check if the input is a string
        if not isinstance(math_expr, str):
            return 'Processing Failed!'
        
        # Remove whitespace for easier validation
        expr_stripped = math_expr.replace(" ", "")
        
        # Define allowed characters for mathematical expressions
        allowed_chars = set('0123456789+-*/().')
        
        # Check if expression contains only allowed characters
        if not all(c in allowed_chars for c in expr_stripped):
            return 'Processing Failed!'
        
        # Additional security checks
        # Check for dangerous patterns
        dangerous_patterns = ['__', 'import', 'exec', 'eval', 'open', 'file', 'input', 'compile']
        for pattern in dangerous_patterns:
            if pattern in math_expr.lower():
                return 'Processing Failed!'
        
        # Check for empty expression
        if not math_expr.strip():
            return 'Processing Failed!'
        
        # Use eval with restricted builtins for safety
        # Create a safe namespace with no builtins
        safe_dict = {
            '__builtins__': {
                'abs': abs,
                'min': min,
                'max': max,
                'sum': sum,
                'round': round
            }
        }
        
        # Evaluate the expression
        result = eval(math_expr, safe_dict, {})
        
        # Convert result to string
        return str(result)
        
    except Exception:
        # Catch any exception (SyntaxError, ZeroDivisionError, etc.)
        return 'Processing Failed!'
