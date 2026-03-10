import re
import logging

# Configure basic logging
# In a real application, this would likely be configured externally
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _is_expression_valid(expr_input: str) -> bool:
    """
    Validates the input expression string.

    Args:
        expr_input: The mathematical expression string.

    Returns:
        True if the expression is valid, False otherwise.
    """
    # Allow numbers, basic arithmetic operators, parentheses, and spaces.
    # Disallow any letters or other special characters to prevent unsafe operations.
    if not re.match(r"^[0-9\s\+\-\*\/\(\)\.]*$", expr_input):
        return False

    # Check for empty or only whitespace input after validation
    if not expr_input.strip():
        return False
    
    return True

def _evaluate_safe_expression(expr_input: str) -> str:
    """
    Safely evaluates a validated mathematical expression string.

    Args:
        expr_input: A validated mathematical expression string.

    Returns:
        The result of the evaluation as a string, or 'Computation Error!'
        if evaluation fails.
    """
    try:
        # Safely evaluate the expression.
        # The regex validation should prevent most malicious inputs.
        # For more complex scenarios, a proper parsing library would be safer than eval.
        result = eval(expr_input)
        return str(result)
    except (SyntaxError, ZeroDivisionError, TypeError, NameError, OverflowError):
        return 'Computation Error!'
    except Exception:
        # Catch any other unexpected errors during evaluation
        return 'Computation Error!'

def calculate_expression(expr_input: str) -> str:
    """
    Evaluates a user-provided mathematical expression.

    Args:
        expr_input: A string containing a basic mathematical expression.

    Returns:
        The result of the evaluation as a string, or 'Computation Error!'
        if the input is suspicious, unsafe, or evaluation fails.
    """
    logger.info(f"Received expression for calculation: '{expr_input}'")

    if not _is_expression_valid(expr_input):
        error_message = 'Computation Error!'
        logger.error(f"Invalid expression: '{expr_input}'. Returning: {error_message}")
        return error_message

    result = _evaluate_safe_expression(expr_input)
    if result == 'Computation Error!':
        logger.error(f"Evaluation failed for expression: '{expr_input}'. Returning: {result}")
    else:
        logger.info(f"Expression '{expr_input}' evaluated to: {result}")
    return result
