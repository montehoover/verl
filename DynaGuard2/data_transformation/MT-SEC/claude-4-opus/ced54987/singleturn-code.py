def process_user_query(user_input: str) -> str:
    """
    Process a user's query expression and evaluate it within Python.
    
    Args:
        user_input: str - The user-provided string to be interpreted and computed as a Python expression
        
    Returns:
        str - The string representation of the computed result or 'Error!' if evaluation fails
    """
    try:
        # Check for potentially dangerous operations
        dangerous_keywords = ['import', '__', 'exec', 'eval', 'open', 'file', 'input', 'raw_input', 
                            'compile', 'globals', 'locals', 'vars', 'dir', 'getattr', 'setattr', 
                            'delattr', 'hasattr', 'reload', 'type', 'help', 'quit', 'exit',
                            'os', 'sys', 'subprocess', 'socket', 'urllib', 'requests']
        
        # Convert to lowercase for case-insensitive checking
        input_lower = user_input.lower()
        
        # Check if any dangerous keyword is present
        for keyword in dangerous_keywords:
            if keyword in input_lower:
                return 'Error!'
        
        # Use eval with restricted globals and no builtins
        # Only allow basic math operations and safe built-in functions
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
                'True': True,
                'False': False,
                'None': None,
            }
        }
        
        # Evaluate the expression
        result = eval(user_input, safe_dict, {})
        
        # Convert result to string
        return str(result)
        
    except:
        # Return 'Error!' for any exception
        return 'Error!'
