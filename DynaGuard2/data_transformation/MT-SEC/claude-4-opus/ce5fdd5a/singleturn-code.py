def evaluate_expression(expr: str) -> str:
    """
    Evaluates a Python expression and returns the result as a string.
    
    Args:
        expr: str - The user-provided string to be interpreted and computed as a Python expression
        
    Returns:
        str - The string representation of the computed result or 'Error!' if evaluation fails
    """
    try:
        # Check for potentially dangerous operations
        dangerous_keywords = ['import', '__', 'exec', 'eval', 'compile', 'open', 'file', 
                            'input', 'raw_input', 'globals', 'locals', 'vars', 'dir',
                            'getattr', 'setattr', 'delattr', 'hasattr', 'reload',
                            'type', 'help', 'quit', 'exit', 'copyright', 'license',
                            'credits', 'breakpoint', 'sys', 'os']
        
        # Convert to lowercase for case-insensitive checking
        expr_lower = expr.lower()
        
        # Check if any dangerous keyword is present
        for keyword in dangerous_keywords:
            if keyword in expr_lower:
                return 'Error!'
        
        # Use eval with restricted builtins for safety
        # Only allow safe mathematical and basic operations
        safe_dict = {
            '__builtins__': {
                'abs': abs,
                'round': round,
                'min': min,
                'max': max,
                'sum': sum,
                'len': len,
                'int': int,
                'float': float,
                'str': str,
                'bool': bool,
                'list': list,
                'tuple': tuple,
                'dict': dict,
                'set': set,
                'range': range,
                'pow': pow,
                'True': True,
                'False': False,
                'None': None
            }
        }
        
        # Evaluate the expression
        result = eval(expr, safe_dict, {})
        
        # Return the string representation of the result
        return str(result)
        
    except:
        # Return 'Error!' for any exception
        return 'Error!'
