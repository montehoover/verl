import logging

# Configure logging to a file in the current directory
logging.basicConfig(filename='quiz_app.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def _is_expression_safe(expr: str) -> bool:
    """
    Checks if the expression is potentially untrustworthy.

    Args:
        expr: str, the user-provided string expression.

    Returns:
        bool, True if the expression is considered safe, False otherwise.
    """
    # Basic check for potentially untrustworthy input.
    # This aims to prevent access to dunder methods/attributes, a common vector.
    # For a production system, more robust sandboxing or parsing would be needed.
    if "__" in expr:
        return False
    return True

def _compute_expression_value(expr: str) -> str:
    """
    Evaluates the expression and returns its string representation or 'Error!'.

    Args:
        expr: str, the user-provided string expression to compute.

    Returns:
        str, the string representation of the computed result or 'Error!'
             if any exception occurs during evaluation.
    """
    try:
        # Evaluate the expression.
        # Note: eval() can be dangerous if the input string is not controlled.
        # The safety check should be done before calling this function.
        # A more secure approach might involve parsing the expression,
        # using ast.literal_eval if applicable, or a more restricted eval environment.
        result = eval(expr)
        return str(result)
    except Exception as e:
        # Catches all exceptions during eval, including SyntaxError, NameError, TypeError, etc.
        logging.error(f"Error computing expression '{expr}': {e}")
        return 'Error!'

def evaluate_expression(expr: str) -> str:
    """
    Processes a string input containing the user's query expression,
    then evaluates it within Python to generate a result.

    Args:
        expr: str, the user-provided string to be interpreted and computed
              as a Python expression.

    Returns:
        str, the string representation of the computed result or an error
             message if evaluation fails or input is deemed untrustworthy.
    """
    logging.info(f"Received expression for evaluation: '{expr}'")

    if not _is_expression_safe(expr):
        logging.warning(f"Expression '{expr}' deemed unsafe. Aborting evaluation.")
        return 'Error!'

    result = _compute_expression_value(expr)

    if result == 'Error!':
        # Specific error is logged in _compute_expression_value
        logging.info(f"Evaluation of '{expr}' resulted in an error.")
    else:
        logging.info(f"Expression '{expr}' evaluated successfully to: '{result}'")

    return result
