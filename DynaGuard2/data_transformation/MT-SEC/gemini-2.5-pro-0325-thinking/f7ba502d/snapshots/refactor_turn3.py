import re
import logging # New import

# Logger configuration constants
EVAL_LOGGER_NAME = 'ExpressionEvaluatorLogger'
LOG_FILE_NAME = 'expression_evaluator.log'

def _setup_logger_if_not_configured():
    """Sets up the logger for expression evaluation if not already configured.

    This function ensures that the logger has handlers and is configured
    to write to the specified log file. It's designed to be called
    before logging messages.
    """
    logger = logging.getLogger(EVAL_LOGGER_NAME)
    if not logger.handlers:  # Configure only if no handlers are present
        logger.setLevel(logging.INFO)
        
        # Create file handler to write logs to a file
        file_handler = logging.FileHandler(LOG_FILE_NAME)
        file_handler.setLevel(logging.INFO)
        
        # Create a formatter for log messages
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add the handler to the logger
        logger.addHandler(file_handler)
    return logger

def safe_eval_expression(expr: str):
    """Safely evaluate a string mathematical expression.

    This function takes a string representing a mathematical expression,
    validates it to ensure it only contains allowed characters (numbers,
    basic arithmetic operators, parentheses, and whitespace), and then
    evaluates it.

    Args:
        expr: The string containing the mathematical expression.
              Example: "1 + 2 * (3 - 1)"

    Returns:
        The numerical result of evaluating the mathematical expression.
        The type of the result will typically be int or float.

    Raises:
        ValueError: If the input expression string contains invalid
                    characters (e.g., letters, disallowed symbols) or
                    if the expression is syntactically incorrect (e.g.,
                    unbalanced parentheses, division by zero).
    """
    logger = _setup_logger_if_not_configured()
    logger.info(f"Attempting to evaluate expression: '{expr}'")

    # Regex to validate the expression:
    # ^ : asserts position at start of the string.
    # [0-9\s\+\-\*\/\(\)\.] : matches any digit, whitespace, plus, minus,
    #                         asterisk, slash, parenthesis, or dot.
    # * : matches the previous token between zero and unlimited times.
    # $ : asserts position at the end of the string.
    # This ensures the entire string consists only of allowed characters.
    allowed_pattern = r"^[0-9\s\+\-\*\/\(\)\.]*$"
    if not re.match(allowed_pattern, expr):
        error_message = (
            "Invalid characters in expression. Only numbers, operators (+, -, *, /), "
            "parentheses, and whitespace are allowed."
        )
        logger.error(f"Validation failed for expression '{expr}': {error_message}")
        raise ValueError(error_message)

    try:
        # Evaluate the sanitized expression.
        # Note: While re.match provides some safety, using eval() directly
        # can still be risky if the sanitization is not perfect or if
        # the environment allows overriding built-ins. For highly secure
        # applications, consider using ast.literal_eval or a dedicated
        # expression parsing library.
        calculated_result = eval(expr)
        logger.info(f"Expression '{expr}' successfully evaluated to: {calculated_result}")
        return calculated_result
    except (SyntaxError, ZeroDivisionError, TypeError, NameError) as e:
        # Catch common errors during evaluation and raise a ValueError.
        logger.error(f"Error evaluating expression '{expr}': {e}", exc_info=True)
        raise ValueError(f"Incorrect expression: {e}")
