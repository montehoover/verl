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
    # Basic check for potentially untrustworthy input.
    # This aims to prevent access to dunder methods/attributes, a common vector.
    # For a production system, more robust sandboxing or parsing would be needed.
    if "__" in expr:
        return 'Error!'

    try:
        # Evaluate the expression.
        # Note: eval() can be dangerous if the input string is not controlled.
        # The "__" check above is a minimal safety measure.
        # A more secure approach might involve parsing the expression,
        # using ast.literal_eval if applicable, or a more restricted eval environment.
        result = eval(expr)
        return str(result)
    except Exception:
        # Catches all exceptions during eval, including SyntaxError, NameError, TypeError, etc.
        return 'Error!'
