def validate_expression(expr: str) -> bool:
    """Validate if the expression is safe to evaluate."""
    forbidden_keywords = ['__', 'import', 'exec', 'eval', 'open', 'file', 'input', 'compile', 'globals', 'locals']
    return not any(forbidden in expr for forbidden in forbidden_keywords)


def create_safe_environment() -> dict:
    """Create a restricted environment for safe evaluation."""
    return {
        '__builtins__': {
            'abs': abs,
            'all': all,
            'any': any,
            'bin': bin,
            'bool': bool,
            'chr': chr,
            'dict': dict,
            'divmod': divmod,
            'enumerate': enumerate,
            'filter': filter,
            'float': float,
            'format': format,
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
            'type': type,
            'zip': zip,
        }
    }


def compute_expression(expr: str, safe_dict: dict) -> str:
    """Compute the expression in a safe environment."""
    result = eval(expr, safe_dict, {})
    return str(result)


def evaluate_expression(expr: str) -> str:
    try:
        # Validate expression
        if not validate_expression(expr):
            return 'Error!'
        
        # Create safe environment
        safe_dict = create_safe_environment()
        
        # Compute and return result
        return compute_expression(expr, safe_dict)
    except:
        return 'Error!'
