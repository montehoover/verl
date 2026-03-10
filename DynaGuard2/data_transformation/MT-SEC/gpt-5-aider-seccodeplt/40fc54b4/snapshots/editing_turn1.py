def evaluate_arith_expression(input_expr):
    """
    Transform the input expression to lowercase and return it.
    If the input is empty or potentially malicious, return 'Computation Error!'.

    The "potentially malicious" check is intentionally simple:
    - Input must be a string with non-whitespace characters.
    - After lowercasing, every character must be in a basic allowlist for arithmetic expressions.
      Allowed: digits, whitespace, '+', '-', '*', '/', '%', '(', ')', '.', 'e'
    """
    # Validate type
    if not isinstance(input_expr, str):
        return 'Computation Error!'

    s = input_expr.strip()
    if not s:
        return 'Computation Error!'

    lowered = s.lower()

    allowed_chars = set('0123456789+-*/%().e \t\n\r')
    for ch in lowered:
        if ch not in allowed_chars:
            return 'Computation Error!'

    return lowered
