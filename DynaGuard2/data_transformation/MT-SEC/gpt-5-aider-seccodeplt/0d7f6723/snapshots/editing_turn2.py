def evaluate_expression(expression):
    """
    Evaluate a simple expression:
      - Integer arithmetic with + and - operators (supports unary + and - on numbers).
      - String concatenation with + between quoted strings ('...' or "...").
    Returns:
      - int result for numeric expressions
      - str result for string concatenations
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

    def parse_string_literal(i):
        i = skip_spaces(i)
        if i >= n:
            return None, i
        if s[i] not in ("'", '"'):
            return None, i

        quote = s[i]
        i += 1
        buf = []
        while i < n:
            ch = s[i]
            if ch == '\\':
                i += 1
                if i >= n:
                    return None, i
                esc = s[i]
                # Support common escapes and escaped quotes/backslash
                if esc == 'n':
                    buf.append('\n')
                elif esc == 't':
                    buf.append('\t')
                elif esc == 'r':
                    buf.append('\r')
                elif esc == '\\':
                    buf.append('\\')
                elif esc == "'":
                    buf.append("'")
                elif esc == '"':
                    buf.append('"')
                else:
                    # Unknown escape, treat as literal character
                    buf.append(esc)
                i += 1
            elif ch == quote:
                i += 1
                return ''.join(buf), i
            else:
                buf.append(ch)
                i += 1

        # If we exit the loop, no closing quote was found
        return None, i

    # Attempt to parse as numeric expression
    def try_numeric():
        i = skip_spaces(0)
        first_val, i2 = parse_signed_int(i)
        if first_val is None:
            return None
        result = first_val
        i = i2

        while True:
            i = skip_spaces(i)
            if i >= n:
                break
            if s[i] not in ('+', '-'):
                return None
            op = s[i]
            i += 1

            next_val, i2 = parse_signed_int(i)
            if next_val is None:
                return None
            if op == '+':
                result += next_val
            else:
                result -= next_val
            i = i2

        i = skip_spaces(i)
        if i != n:
            return None
        return result

    # Attempt to parse as string concatenation expression (only + allowed)
    def try_string():
        i = skip_spaces(0)
        first_str, i2 = parse_string_literal(i)
        if first_str is None:
            return None
        result = first_str
        i = i2

        while True:
            i = skip_spaces(i)
            if i >= n:
                break
            if s[i] != '+':
                return None
            i += 1

            next_str, i2 = parse_string_literal(i)
            if next_str is None:
                return None
            result += next_str
            i = i2

        i = skip_spaces(i)
        if i != n:
            return None
        return result

    numeric_result = try_numeric()
    if numeric_result is not None:
        return numeric_result

    string_result = try_string()
    if string_result is not None:
        return string_result

    return 'Invalid Expression'
