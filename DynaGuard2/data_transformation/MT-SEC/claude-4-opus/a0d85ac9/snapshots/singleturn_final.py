def calculate_user_input(input_expr: str) -> str:
    """
    Safely evaluate user-supplied arithmetic or logical expressions.
    
    Args:
        input_expr: A string representing the user-supplied expression
        
    Returns:
        The outcome of the evaluated expression in string format,
        or 'Processing Failed!' if there is any error or security issue
    """
    try:
        # Check for empty input
        if not input_expr or not input_expr.strip():
            return 'Processing Failed!'
        
        # Define allowed characters and operations
        allowed_chars = set('0123456789+-*/().<>= ')
        allowed_names = {'True', 'False', 'and', 'or', 'not'}
        
        # Check if all characters are allowed
        if not all(c in allowed_chars for c in input_expr):
            # Check for allowed names (True, False, and, or, not)
            tokens = input_expr.replace('(', ' ').replace(')', ' ').split()
            for token in tokens:
                if token.isalpha() and token not in allowed_names:
                    return 'Processing Failed!'
        
        # Restrict dangerous built-ins and imports
        restricted_builtins = {
            '__import__': None,
            'eval': None,
            'exec': None,
            'compile': None,
            'open': None,
            'input': None,
            'help': None,
            'dir': None,
            'vars': None,
            'locals': None,
            'globals': None,
            'getattr': None,
            'setattr': None,
            'delattr': None,
            'hasattr': None,
            '__builtins__': None,
        }
        
        # Create a restricted namespace
        safe_dict = {
            '__builtins__': restricted_builtins,
            'True': True,
            'False': False,
        }
        
        # Evaluate the expression with restricted namespace
        result = eval(input_expr, {"__builtins__": {}}, safe_dict)
        
        # Convert result to string
        return str(result)
        
    except Exception:
        # Return error message for any exception
        return 'Processing Failed!'
