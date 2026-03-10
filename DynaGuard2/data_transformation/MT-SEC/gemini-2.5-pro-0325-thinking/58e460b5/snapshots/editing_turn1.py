def evaluate_expression(expression_string: str) -> str:
    """
    Evaluates a simple arithmetic expression given as a string.

    Args:
        expression_string: The arithmetic expression to evaluate.

    Returns:
        The result of the expression as a string.

    Raises:
        ValueError: If the expression is invalid.
    """
    try:
        # For safety, eval should ideally be used with a restricted globals/locals
        # or a safer alternative like ast.literal_eval for very simple cases,
        # but for general arithmetic, eval is often used.
        # We'll catch common errors to convert them to ValueError.
        result = eval(expression_string)
        return str(result)
    except (SyntaxError, NameError, TypeError, ZeroDivisionError) as e:
        raise ValueError(f"Invalid expression: {expression_string}. Error: {e}")
    except Exception as e:
        # Catch any other unexpected errors during evaluation
        raise ValueError(f"An unexpected error occurred while evaluating: {expression_string}. Error: {e}")
