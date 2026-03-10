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
    # Basic safety check: disallow expressions containing double underscores
    # to prevent access to many internal attributes and methods.
    if "__" in quiz_expr:
        return "Error!"

    try:
        # Evaluate the expression in a restricted environment.
        # globals: {'__builtins__': {}} - This removes access to all built-in functions.
        # locals: {} - No local variables are predefined.
        # This means only literals (numbers, strings) and basic operators can be used.
        result = eval(quiz_expr, {'__builtins__': {}}, {})
        return str(result)
    except Exception:
        # Catches all exceptions, including SyntaxError, NameError, TypeError,
        # ZeroDivisionError, etc., that may occur during parsing or evaluation.
        return "Error!"
