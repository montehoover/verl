import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('MINDTEK.Calculator')


def validate_expression(calc_string: str) -> bool:
    """Validate that the expression is safe to evaluate."""
    # Check for dangerous operations
    dangerous_keywords = ['import', '__', 'exec', 'eval', 'compile', 'open', 'file', 'input', 'raw_input', 'globals', 'locals', 'vars', 'dir', 'getattr', 'setattr', 'delattr', 'hasattr', 'type', 'isinstance', 'issubclass', 'callable', 'classmethod', 'staticmethod', 'property', 'super', 'object', 'print', 'help', 'quit', 'exit', 'copyright', 'credits', 'license']
    
    for keyword in dangerous_keywords:
        if keyword in calc_string.lower():
            logger.warning(f"Dangerous keyword '{keyword}' detected in expression: {calc_string}")
            return False
    
    # Check for function calls
    if '(' in calc_string and ')' in calc_string:
        logger.warning(f"Function call detected in expression: {calc_string}")
        return False
    
    # Only allow numbers, basic operators, spaces, and decimal points
    allowed_chars = '0123456789+-*/ .'
    for char in calc_string:
        if char not in allowed_chars:
            logger.warning(f"Invalid character '{char}' detected in expression: {calc_string}")
            return False
    
    logger.debug(f"Expression validated successfully: {calc_string}")
    return True


def evaluate_expression(calc_string: str) -> str:
    """Evaluate a validated mathematical expression."""
    try:
        result = eval(calc_string)
        logger.debug(f"Expression evaluated: {calc_string} = {result}")
        return str(result)
    except Exception as e:
        logger.error(f"Failed to evaluate expression '{calc_string}': {str(e)}")
        raise ValueError("Failed to evaluate expression")


def exec_calculation(calc_string: str) -> str:
    logger.info(f"Calculation attempt started for: {calc_string}")
    
    try:
        if not validate_expression(calc_string):
            logger.error(f"Validation failed for expression: {calc_string}")
            return 'Computation Error!'
        
        result = evaluate_expression(calc_string)
        logger.info(f"Calculation successful: {calc_string} = {result}")
        return result
        
    except Exception as e:
        logger.error(f"Calculation failed for '{calc_string}': {str(e)}")
        return 'Computation Error!'
