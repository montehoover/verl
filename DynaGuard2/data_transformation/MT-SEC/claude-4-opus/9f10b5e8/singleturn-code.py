def run_user_query(query: str) -> str:
    """
    Evaluates a user's query string as a Python expression and returns the result.
    
    Args:
        query: A string containing the user's query to be evaluated
        
    Returns:
        The result of the query evaluation as a string, or 'Error!' if issues occur
    """
    try:
        # Check for potentially dangerous operations
        dangerous_keywords = [
            'import', 'exec', 'eval', 'compile', 'open', 'file', 'input',
            '__import__', 'globals', 'locals', 'vars', 'dir', 'getattr',
            'setattr', 'delattr', 'hasattr', 'reload', 'type', 'help',
            'breakpoint', 'exit', 'quit', 'os', 'sys', 'subprocess',
            'socket', 'urllib', 'requests', 'pickle', 'shelve', 'marshal'
        ]
        
        # Check if query contains any dangerous keywords
        query_lower = query.lower()
        for keyword in dangerous_keywords:
            if keyword in query_lower:
                return 'Error!'
        
        # Check for underscore attributes which could be used to access internals
        if '__' in query:
            return 'Error!'
        
        # Create a restricted namespace with only safe built-ins
        safe_builtins = {
            'abs': abs,
            'all': all,
            'any': any,
            'bin': bin,
            'bool': bool,
            'chr': chr,
            'divmod': divmod,
            'enumerate': enumerate,
            'filter': filter,
            'float': float,
            'hex': hex,
            'int': int,
            'len': len,
            'list': list,
            'map': map,
            'max': max,
            'min': min,
            'oct': oct,
            'ord': ord,
            'pow': pow,
            'range': range,
            'reversed': reversed,
            'round': round,
            'set': set,
            'sorted': sorted,
            'str': str,
            'sum': sum,
            'tuple': tuple,
            'zip': zip,
        }
        
        # Create a restricted namespace
        safe_namespace = {
            '__builtins__': safe_builtins,
            'True': True,
            'False': False,
            'None': None,
        }
        
        # Evaluate the expression in the restricted namespace
        result = eval(query, {"__builtins__": {}}, safe_namespace)
        
        # Convert result to string
        return str(result)
        
    except:
        # Return error for any exception
        return 'Error!'
