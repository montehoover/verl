import datetime
import os


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


def log_expression(expr: str, result: str) -> None:
    """Log expression and result to a file."""
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] Expression: {expr!r} | Result: {result}\n"
        
        log_file = os.path.join(os.getcwd(), "expression_log.txt")
        with open(log_file, 'a') as f:
            f.write(log_entry)
    except:
        pass  # Silently fail if logging doesn't work


def compute_expression(expr: str, safe_dict: dict) -> str:
    """Compute the expression in a safe environment."""
    result = eval(expr, safe_dict, {})
    return str(result)


def evaluate_expression(expr: str) -> str:
    # Guard clause for validation
    if not validate_expression(expr):
        log_expression(expr, 'Error!')
        return 'Error!'
    
    # Try to compute expression
    try:
        safe_dict = create_safe_environment()
        result = compute_expression(expr, safe_dict)
        log_expression(expr, result)
        return result
    except:
        log_expression(expr, 'Error!')
        return 'Error!'
