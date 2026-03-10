def evaluate_quiz_expression(quiz_expr: str) -> str:
    """
    Evaluates a Python expression and returns the result as a string.
    
    Args:
        quiz_expr: A Python expression provided by the user for evaluation
        
    Returns:
        The result of the evaluation as a string or 'Error!' if evaluation fails
    """
    try:
        # Check for dangerous operations
        dangerous_keywords = [
            '__import__', 'import', 'exec', 'eval', 'compile', 
            'open', 'file', 'input', 'raw_input', 'execfile',
            'globals', 'locals', 'vars', 'dir', 'getattr', 
            'setattr', 'delattr', 'hasattr', '__builtins__',
            'breakpoint', 'help', 'quit', 'exit', 'copyright',
            'license', 'credits', '__', 'lambda', 'def', 'class',
            'with', 'as', 'assert', 'del', 'from', 'global',
            'nonlocal', 'pass', 'raise', 'return', 'yield',
            'try', 'except', 'finally', 'while', 'for', 'if',
            'elif', 'else', 'break', 'continue', 'and', 'or',
            'not', 'in', 'is', 'None', 'True', 'False'
        ]
        
        # Check if any dangerous keyword is in the expression
        for keyword in dangerous_keywords:
            if keyword in quiz_expr:
                return 'Error!'
        
        # Create a restricted namespace
        safe_dict = {
            '__builtins__': {
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
        }
        
        # Evaluate the expression
        result = eval(quiz_expr, {"__builtins__": {}}, safe_dict)
        return str(result)
        
    except:
        return 'Error!'
