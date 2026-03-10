import logging

# Configure logging
log_file = "quiz_evaluations.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        # logging.StreamHandler() # Uncomment to also log to console
    ]
)

def _is_safe_expression(quiz_expr: str) -> bool:
    """
    Checks if the expression is safe to evaluate.
    Currently, it only checks for the presence of double underscores.
    """
    if "__" in quiz_expr:
        return False
    return True

def _evaluate_expression_safely(quiz_expr: str) -> any:
    """
    Evaluates the expression in a restricted environment.
    Raises an exception if evaluation fails.
    """
    # Evaluate the expression in a restricted environment.
    # globals: {'__builtins__': {}} - This removes access to all built-in functions.
    # locals: {} - No local variables are predefined.
    # This means only literals (numbers, strings) and basic operators can be used.
    return eval(quiz_expr, {'__builtins__': {}}, {})

def evaluate_quiz_expression(quiz_expr: str) -> str:
    """
    Evaluates a quiz expression string and returns the result.

    Args:
        quiz_expr: A string representing a Python expression provided by the user.

    Returns:
        The result of the evaluation as a string, or 'Error!'
        if the input is invalid, unsafe to process, or evaluation fails
        with an exception.
    """
    logging.info(f"Received expression for evaluation: '{quiz_expr}'")

    if not isinstance(quiz_expr, str): # Guard clause for type
        logging.error(f"Invalid input type for expression: '{quiz_expr}'. Expected string.")
        return "Error!"

    if not _is_safe_expression(quiz_expr): # Guard clause for safety
        logging.warning(f"Unsafe expression detected: '{quiz_expr}'")
        return "Error!"

    try:
        result = _evaluate_expression_safely(quiz_expr)
        final_result_str = str(result)
        logging.info(f"Expression '{quiz_expr}' evaluated successfully. Result: {final_result_str}")
        return final_result_str
    except Exception as e:
        # Catches all exceptions from _evaluate_expression_safely,
        # including SyntaxError, NameError, TypeError, ZeroDivisionError, etc.
        logging.error(f"Error evaluating expression '{quiz_expr}': {e}", exc_info=True)
        return "Error!"
