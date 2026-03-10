def evaluate_expression(expression):
    """
    Evaluate a simple arithmetic expression containing integers with + and - operators.
    Returns:
      - int result if valid
      - 'Invalid Expression' string if invalid
    """
    if not isinstance(expression, str):
        return 'Invalid Expression'

    s = expression
    n = len(s)

    def skip_spaces(i):
        while i < n and s[i].isspace():
            i += 1
        return i

    def parse_signed_int(i):
        i = skip_spaces(i)
        if i >= n:
            return None, i

        sign = 1
        if s[i] == '+' or s[i] == '-':
            if s[i] == '-':
                sign = -1
            i += 1

        start_digits = i
        while i < n and s[i].isdigit():
            i += 1

        if i == start_digits:
            return None, start_digits  # no digits after optional sign

        value = int(s[start_digits:i]) * sign
        return value, i

    i = 0
    i = skip_spaces(i)

    # Parse first number (with optional sign)
    first_val, i2 = parse_signed_int(i)
    if first_val is None:
        return 'Invalid Expression'
    result = first_val
    i = i2

    while True:
        i = skip_spaces(i)
        if i >= n:
            break  # end of expression

        # Expect operator
        if s[i] not in ('+', '-'):
            return 'Invalid Expression'
        op = s[i]
        i += 1

        # Parse next signed integer
        next_val, i2 = parse_signed_int(i)
        if next_val is None:
            return 'Invalid Expression'
        if op == '+':
            result += next_val
        else:
            result -= next_val
        i = i2

    # Ensure nothing but spaces remain
    i = skip_spaces(i)
    if i != n:
        return 'Invalid Expression'

    return result
