import re

def validate_expression(expr: str) -> bool:
    if expr is None:
        return False
    return bool(re.fullmatch(r'(?=.*\d)[\d+\-*/\s]+', expr))

def parse_expression(expr: str):
    """
    Parse a validated arithmetic expression into a list of numbers (ints) and operator strings.
    Supports unary + and - for numbers. Spaces are ignored.
    """
    if expr is None:
        return []
    tokens = []
    i = 0
    n = len(expr)
    prev_is_op = True  # At start, unary signs are allowed

    while i < n:
        ch = expr[i]
        if ch.isspace():
            i += 1
            continue

        if ch in '+-':
            if prev_is_op:
                j = i + 1
                while j < n and expr[j].isspace():
                    j += 1
                if j < n and expr[j].isdigit():
                    sign = -1 if ch == '-' else 1
                    k = j
                    while k < n and expr[k].isdigit():
                        k += 1
                    tokens.append(sign * int(expr[j:k]))
                    i = k
                    prev_is_op = False
                    continue
            tokens.append(ch)
            i += 1
            prev_is_op = True
            continue

        if ch in '*/':
            tokens.append(ch)
            i += 1
            prev_is_op = True
            continue

        if ch.isdigit():
            j = i
            while j < n and expr[j].isdigit():
                j += 1
            tokens.append(int(expr[i:j]))
            i = j
            prev_is_op = False
            continue

        # Skip any unexpected character (shouldn't happen if validated)
        i += 1

    return tokens
