import re
import logging


# Configure logger for this module
logger = logging.getLogger(__name__)


def evaluate_user_expression(expression: str):
    """
    Safely evaluate a mathematical expression provided by the user.
    
    This function takes a string containing a mathematical expression and
    evaluates it safely, ensuring that only basic mathematical operations
    are allowed (addition, subtraction, multiplication, division, and parentheses).
    
    Args:
        expression (str): The string containing the user's mathematical expression.
        
    Returns:
        float or int: The result of evaluating the mathematical expression.
        
    Raises:
        ValueError: If the input contains invalid characters or if the 
                   expression is syntactically incorrect.
                   
    Examples:
        >>> evaluate_user_expression("2 + 3 * 4")
        14
        >>> evaluate_user_expression("(10 - 5) / 2")
        2.5
    """
    # Log the incoming expression
    logger.info(f"Evaluating expression: '{expression}'")
    
    # Remove all whitespace from the expression for easier processing
    cleaned_expression = expression.replace(" ", "")
    
    # Validate that the expression is not empty
    if not cleaned_expression:
        logger.error("Empty expression provided")
        raise ValueError("Empty expression")
    
    # Check for invalid characters using regex
    # Only allow digits, basic operators (+, -, *, /), parentheses, and decimal points
    if not re.match(r'^[0-9+\-*/().\s]+$', cleaned_expression):
        logger.error(f"Invalid characters detected in expression: '{expression}'")
        raise ValueError("Invalid characters in expression")
    
    try:
        # Evaluate the expression with a restricted namespace
        # This prevents access to built-in functions and makes eval safer
        result = eval(cleaned_expression, {"__builtins__": {}}, {})
        
        # Log successful evaluation
        logger.info(f"Expression '{expression}' evaluated successfully. Result: {result}")
        
        return result
        
    except (SyntaxError, ZeroDivisionError, TypeError) as e:
        # Log the error
        logger.error(f"Failed to evaluate expression '{expression}': {str(e)}")
        
        # Convert any evaluation errors into ValueError with descriptive message
        raise ValueError(f"Invalid expression: {str(e)}")
