import logging
from datetime import datetime
import os

# Configure logging
def _setup_logging():
    """Sets up logging configuration for quiz expression evaluations."""
    log_filename = os.path.join(os.getcwd(), 'quiz_evaluations.log')
    
    # Create a custom formatter for human-readable logs
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_filename, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # Get logger
    logger = logging.getLogger('quiz_evaluator')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers and add new one
    logger.handlers.clear()
    logger.addHandler(file_handler)
    
    return logger

# Initialize logger
_logger = _setup_logging()


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
    # Log the incoming expression
    _logger.info(f"Evaluation started - Expression: '{quiz_expr}'")
    
    # Guard clause: check if expression is safe
    if not _is_expression_safe(quiz_expr):
        _logger.warning(f"Unsafe expression detected - Expression: '{quiz_expr}' - Result: 'Error!'")
        return 'Error!'
    
    # Guard clause: handle empty or None input
    if not quiz_expr or quiz_expr.strip() == '':
        _logger.warning(f"Empty or invalid expression - Expression: '{quiz_expr}' - Result: 'Error!'")
        return 'Error!'
    
    try:
        safe_namespace = _get_safe_namespace()
        result = _evaluate_safe_expression(quiz_expr, safe_namespace)
        _logger.info(f"Evaluation successful - Expression: '{quiz_expr}' - Result: '{result}'")
        return result
    except Exception as e:
        _logger.error(f"Evaluation failed - Expression: '{quiz_expr}' - Error: '{type(e).__name__}: {str(e)}' - Result: 'Error!'")
        return 'Error!'
