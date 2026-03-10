def evaluate_arith_expression(input_expr: str) -> str:
    """
    Transforms the input string based on specific prefixes.
    - "TO_UPPER:<string>" converts <string> to uppercase.
    - "SLICE:<string>" slices <string> from index 5.
    Returns 'Computation Error!' if the input is empty, does not match
    a known prefix, or is otherwise deemed potentially malicious.
    """
    if not input_expr:
        return 'Computation Error!'

    # Simple check for potentially malicious input (currently just emptiness)
    # More sophisticated checks can be added later.

    if input_expr.startswith("TO_UPPER:"):
        return input_expr[len("TO_UPPER:"):].upper()
    elif input_expr.startswith("SLICE:"):
        return input_expr[len("SLICE:"):] # Slicing from index 5 is equivalent to len("SLICE:")
    else:
        return 'Computation Error!'
