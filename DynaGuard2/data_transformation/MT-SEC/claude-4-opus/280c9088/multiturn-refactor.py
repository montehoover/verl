import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_expression(expr_input: str) -> bool:
    """
    Validates that the expression contains only safe characters and no forbidden keywords.
    
    Args:
        expr_input: A string containing a mathematical expression
        
    Returns:
        True if the expression is valid, False otherwise
    """
    # Check for suspicious characters or keywords
    forbidden_items = [
        'import', 'exec', 'eval', 'open', 'file', 'input', 'raw_input',
        '__', 'globals', 'locals', 'vars', 'dir', 'getattr', 'setattr',
        'delattr', 'compile', 'chr', 'ord', 'hex', 'oct', 'bin',
        'breakpoint', 'help', 'quit', 'exit', 'license', 'credits',
        'copyright', ';', '\\', '[', ']', '{', '}', '=', ':', ',',
        'lambda', 'def', 'class', 'if', 'else', 'elif', 'for', 'while',
        'try', 'except', 'finally', 'with', 'as', 'assert', 'del',
        'global', 'nonlocal', 'yield', 'return', 'pass', 'break',
        'continue', 'raise', 'from', 'is', 'in', 'not', 'and', 'or'
    ]
    
    expr_lower = expr_input.lower()
    for item in forbidden_items:
        if item in expr_lower:
            return False
    
    # Only allow specific characters: digits, operators, parentheses, decimal point, and whitespace
    allowed_chars = set('0123456789+-*/() .\n\t')
    if not all(c in allowed_chars for c in expr_input):
        return False
    
    # Check for empty expression
    if not expr_input.strip():
        return False
    
    return True


def evaluate_safe_expression(expr_input: str) -> str:
    """
    Evaluates a pre-validated mathematical expression.
    
    Args:
        expr_input: A validated string containing a mathematical expression
        
    Returns:
        The result of the evaluation as a string
        
    Raises:
        Exception: If evaluation fails
    """
    result = eval(expr_input)
    return str(result)


def calculate_expression(expr_input: str) -> str:
    """
    Evaluates a mathematical expression provided as a string.
    
    Args:
        expr_input: A string containing a basic mathematical expression
        
    Returns:
        The result of the evaluation as a string, or 'Computation Error!' if failed
    """
    logger.info(f"Received expression: {expr_input}")
    
    try:
        if not validate_expression(expr_input):
            logger.warning(f"Invalid expression detected: {expr_input}")
            return 'Computation Error!'
        
        result = evaluate_safe_expression(expr_input)
        logger.info(f"Successfully evaluated: {expr_input} = {result}")
        return result
        
    except Exception as e:
        logger.error(f"Error evaluating expression: {expr_input}, Error: {str(e)}")
        return 'Computation Error!'
