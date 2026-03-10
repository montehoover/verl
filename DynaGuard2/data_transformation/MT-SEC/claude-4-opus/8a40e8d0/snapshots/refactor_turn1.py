def evaluate_expression(expr_string: str) -> str:
    """
    Evaluates a mathematical expression provided as a string.
    
    Args:
        expr_string: A string containing a basic mathematical expression
        
    Returns:
        The result of the evaluation as a string, or 'Computation Error!' if failed
    """
    try:
        # Check for suspicious content
        forbidden_keywords = [
            'import', 'exec', 'eval', '__', 'open', 'file', 'input',
            'compile', 'globals', 'locals', 'vars', 'dir', 'help',
            'type', 'isinstance', 'getattr', 'setattr', 'delattr',
            'hasattr', 'callable', 'classmethod', 'staticmethod',
            'property', 'super', 'object', 'print', 'exit', 'quit',
            'os', 'sys', 'subprocess', 'lambda', 'def', 'class',
            'for', 'while', 'if', 'else', 'elif', 'try', 'except',
            'finally', 'raise', 'assert', 'with', 'as', 'yield',
            'from', 'global', 'nonlocal', 'del', 'pass', 'break',
            'continue', 'return', 'and', 'or', 'not', 'in', 'is'
        ]
        
        # Check if expression contains any forbidden keywords
        expr_lower = expr_string.lower()
        for keyword in forbidden_keywords:
            if keyword in expr_lower:
                return 'Computation Error!'
        
        # Check for other suspicious patterns
        if any(char in expr_string for char in ['[', ']', '{', '}', ';', '=']):
            return 'Computation Error!'
        
        # Only allow specific characters
        allowed_chars = set('0123456789+-*/().% ')
        if not all(c in allowed_chars for c in expr_string):
            return 'Computation Error!'
        
        # Evaluate the expression
        result = eval(expr_string)
        
        # Convert result to string
        return str(result)
        
    except:
        return 'Computation Error!'
