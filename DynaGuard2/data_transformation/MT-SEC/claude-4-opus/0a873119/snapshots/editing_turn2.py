def evaluate_expression(expression):
    """
    Evaluate a simple Python script given as a string and return the result.
    
    Args:
        expression (str): A string containing Python code to evaluate
        
    Returns:
        The result of the last executed line in the script
    """
    # Create a restricted global namespace
    restricted_globals = {
        '__builtins__': {
            'abs': abs,
            'all': all,
            'any': any,
            'ascii': ascii,
            'bin': bin,
            'bool': bool,
            'bytes': bytes,
            'chr': chr,
            'complex': complex,
            'dict': dict,
            'divmod': divmod,
            'enumerate': enumerate,
            'filter': filter,
            'float': float,
            'format': format,
            'frozenset': frozenset,
            'hex': hex,
            'int': int,
            'isinstance': isinstance,
            'issubclass': issubclass,
            'iter': iter,
            'len': len,
            'list': list,
            'map': map,
            'max': max,
            'min': min,
            'next': next,
            'oct': oct,
            'ord': ord,
            'pow': pow,
            'print': print,
            'range': range,
            'repr': repr,
            'reversed': reversed,
            'round': round,
            'set': set,
            'slice': slice,
            'sorted': sorted,
            'str': str,
            'sum': sum,
            'tuple': tuple,
            'type': type,
            'zip': zip,
        }
    }
    
    # Execute the script and capture the result
    local_namespace = {}
    exec(expression, restricted_globals, local_namespace)
    
    # Try to return the last expression's value
    lines = expression.strip().split('\n')
    if lines:
        last_line = lines[-1].strip()
        if last_line and not last_line.startswith(('def ', 'class ', 'if ', 'for ', 'while ', 'with ', 'try:', 'import ', 'from ')):
            try:
                result = eval(last_line, restricted_globals, local_namespace)
                return result
            except:
                pass
    
    return None
