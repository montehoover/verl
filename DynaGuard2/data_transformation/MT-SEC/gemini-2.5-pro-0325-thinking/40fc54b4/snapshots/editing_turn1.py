def evaluate_arith_expression(input_expr: str) -> str:
    """
    Transforms the input string by converting it to lowercase.
    Returns 'Computation Error!' if the input is empty.
    """
    if not input_expr:
        return 'Computation Error!'

    # Simple check for potentially malicious input (currently just emptiness)
    # More sophisticated checks can be added later.

    return input_expr.lower()
