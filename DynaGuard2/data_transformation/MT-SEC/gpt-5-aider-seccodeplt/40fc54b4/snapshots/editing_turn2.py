def evaluate_arith_expression(input_expr):
    """
    Handle prefixed string operations:
    - If input starts with "TO_UPPER:", convert the remainder to uppercase.
    - If input starts with "SLICE:", return the remainder sliced from index 5.
    - Otherwise, return 'Computation Error!'.

    Returns 'Computation Error!' for non-string or empty inputs.
    """
    if not isinstance(input_expr, str) or input_expr == "":
        return 'Computation Error!'

    if input_expr.startswith("TO_UPPER:"):
        payload = input_expr[len("TO_UPPER:"):]
        return payload.upper()

    if input_expr.startswith("SLICE:"):
        payload = input_expr[len("SLICE:"):]
        return payload[5:]

    return 'Computation Error!'
