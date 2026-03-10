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
    if not isinstance(quiz_expr, str): # Guard clause for type
        return "Error!"

    if not _is_safe_expression(quiz_expr): # Guard clause for safety
        return "Error!"

    try:
        result = _evaluate_expression_safely(quiz_expr)
        return str(result)
    except Exception:
        # Catches all exceptions from _evaluate_expression_safely,
        # including SyntaxError, NameError, TypeError, ZeroDivisionError, etc.
        return "Error!"
