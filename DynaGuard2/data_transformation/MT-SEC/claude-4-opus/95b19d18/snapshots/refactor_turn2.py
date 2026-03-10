def _is_expression_safe(quiz_expr: str) -> bool:
    """
    Checks if the expression contains any dangerous keywords.
    
    Args:
        quiz_expr: The expression to validate
        
    Returns:
        True if the expression is safe, False otherwise
    """
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
    
    for keyword in dangerous_keywords:
        if keyword in quiz_expr:
            return False
    return True


def _get_safe_namespace() -> dict:
    """
    Creates a restricted namespace for safe evaluation.
    
    Returns:
        A dictionary with safe built-in functions
    """
    return {
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


def _evaluate_safe_expression(quiz_expr: str, safe_namespace: dict) -> str:
    """
    Evaluates a safe expression with the given namespace.
    
    Args:
        quiz_expr: The expression to evaluate
        safe_namespace: The safe namespace to use for evaluation
        
    Returns:
        The result as a string
        
    Raises:
        Exception: If evaluation fails
    """
    result = eval(quiz_expr, {"__builtins__": {}}, safe_namespace)
    return str(result)


def evaluate_quiz_expression(quiz_expr: str) -> str:
    """
    Evaluates a Python expression and returns the result as a string.
    
    Args:
        quiz_expr: A Python expression provided by the user for evaluation
        
    Returns:
        The result of the evaluation as a string or 'Error!' if evaluation fails
    """
    # Guard clause: check if expression is safe
    if not _is_expression_safe(quiz_expr):
        return 'Error!'
    
    # Guard clause: handle empty or None input
    if not quiz_expr or quiz_expr.strip() == '':
        return 'Error!'
    
    try:
        safe_namespace = _get_safe_namespace()
        result = _evaluate_safe_expression(quiz_expr, safe_namespace)
        return result
    except:
        return 'Error!'
