def exec_calculation(calc_string: str) -> str:
    """
    Safely evaluates a mathematical expression provided as a string.
    
    Args:
        calc_string: A string containing a mathematical expression to evaluate
        
    Returns:
        The result of the evaluation as a string, or 'Computation Error!' if failed
    """
    try:
        # List of allowed names in the expression
        allowed_names = {
            'abs', 'round', 'min', 'max', 'sum', 'pow', 'int', 'float'
        }
        
        # Check if the string is empty or None
        if not calc_string or not calc_string.strip():
            return 'Computation Error!'
        
        # Create a restricted namespace for evaluation
        safe_dict = {
            '__builtins__': {},
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'sum': sum,
            'pow': pow,
            'int': int,
            'float': float
        }
        
        # Use compile to check if the code is an expression (not statements)
        code = compile(calc_string, '<string>', 'eval')
        
        # Check for potentially harmful names
        import ast
        tree = ast.parse(calc_string, mode='eval')
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id not in allowed_names:
                # Check if it's not a number
                try:
                    float(node.id)
                except ValueError:
                    return 'Computation Error!'
            elif isinstance(node, ast.Attribute):
                # Disallow attribute access
                return 'Computation Error!'
            elif isinstance(node, ast.Call):
                # Only allow calls to allowed functions
                if isinstance(node.func, ast.Name) and node.func.id not in allowed_names:
                    return 'Computation Error!'
        
        # Evaluate the expression in the restricted environment
        result = eval(code, safe_dict)
        
        # Convert the result to string
        return str(result)
        
    except Exception:
        return 'Computation Error!'
