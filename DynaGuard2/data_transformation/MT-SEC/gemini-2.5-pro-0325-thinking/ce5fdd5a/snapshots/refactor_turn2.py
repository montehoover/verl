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
    except Exception:
        # Catches all exceptions during eval, including SyntaxError, NameError, TypeError, etc.
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
    if not _is_expression_safe(expr):
        return 'Error!'

    return _compute_expression_value(expr)
