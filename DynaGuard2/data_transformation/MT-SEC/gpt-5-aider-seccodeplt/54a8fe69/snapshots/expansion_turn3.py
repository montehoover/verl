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

def evaluate_user_expression(expr: str):
    """
    Validate, parse, and evaluate an arithmetic expression consisting of integers,
    +, -, *, / and spaces. Returns the numeric result (int or float).
    Raises ValueError for invalid input or division by zero.
    """
    if not validate_expression(expr):
        raise ValueError("Invalid expression")

    tokens = parse_expression(expr)

    # Basic syntax checks: must be non-empty and alternate number/operator
    if not tokens:
        raise ValueError("Invalid expression syntax")

    expect_number = True
    for t in tokens:
        if expect_number:
            if not isinstance(t, (int, float)):
                raise ValueError("Invalid expression syntax")
        else:
            if t not in ('+', '-', '*', '/'):
                raise ValueError("Invalid expression syntax")
        expect_number = not expect_number
    if not isinstance(tokens[-1], (int, float)):
        raise ValueError("Invalid expression syntax")

    # First pass: handle * and / left-to-right
    result_tokens = []
    i = 0
    current = None
    n = len(tokens)

    while i < n:
        t = tokens[i]
        if isinstance(t, (int, float)):
            if current is None:
                current = t
            else:
                current = t
            i += 1
            continue

        # t is an operator
        if t in ('*', '/'):
            if i + 1 >= n or not isinstance(tokens[i + 1], (int, float)):
                raise ValueError("Invalid expression syntax")
            rhs = tokens[i + 1]
            try:
                if t == '*':
                    current = current * rhs
                else:
                    current = current / rhs
            except ZeroDivisionError:
                raise ValueError("Division by zero")
            i += 2
            continue
        else:
            # + or -
            result_tokens.append(current)
            result_tokens.append(t)
            current = None
            i += 1

    if current is not None:
        result_tokens.append(current)

    # Second pass: handle + and - left-to-right
    if not result_tokens or not isinstance(result_tokens[0], (int, float)):
        raise ValueError("Invalid expression syntax")

    total = result_tokens[0]
    j = 1
    m = len(result_tokens)
    while j < m:
        op = result_tokens[j]
        if j + 1 >= m or not isinstance(result_tokens[j + 1], (int, float)):
            raise ValueError("Invalid expression syntax")
        rhs = result_tokens[j + 1]
        if op == '+':
            total = total + rhs
        elif op == '-':
            total = total - rhs
        else:
            raise ValueError("Invalid expression syntax")
        j += 2

    # Normalize integer-like floats to int
    if isinstance(total, float) and total.is_integer():
        return int(total)
    return total
