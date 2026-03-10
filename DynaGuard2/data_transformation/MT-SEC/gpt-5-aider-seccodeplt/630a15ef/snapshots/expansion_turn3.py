import re

_ALLOWED_PATTERN = re.compile(r'[0-9+\-*/ ]+')

def is_safe_query(query: str) -> bool:
    """
    Returns True if the input string contains only digits, +, -, *, /, and spaces.
    Uses regex for validation.
    """
    if not isinstance(query, str):
        return False
    return bool(_ALLOWED_PATTERN.fullmatch(query))


def evaluate_expression(expr: str):
    """
    Parses and evaluates a string expression containing integers and the operators
    +, -, *, / with standard operator precedence. Spaces are allowed.
    Raises ValueError for unsupported characters or malformed expressions.
    """
    if not isinstance(expr, str):
        raise ValueError("Expression must be a string")

    if not _ALLOWED_PATTERN.fullmatch(expr):
        raise ValueError("Unsupported characters in expression")

    s = expr.strip()
    if not s:
        raise ValueError("Malformed expression")

    i = 0
    n = len(s)
    values = []  # stores numbers, with + and - applied as sign
    pending_op = None  # '*' or '/' waiting for right operand
    expect_number = True  # whether we are expecting a number next
    sign = 1  # accumulated unary sign for the next number when handling +/-

    while i < n:
        ch = s[i]

        # Skip spaces
        if ch == ' ':
            i += 1
            continue

        if expect_number:
            # Handle unary plus/minus
            if ch == '+':
                sign *= 1
                i += 1
                continue
            if ch == '-':
                sign *= -1
                i += 1
                continue

            # Parse number
            if ch.isdigit():
                j = i
                while j < n and s[j].isdigit():
                    j += 1
                num = int(s[i:j]) * sign
                sign = 1  # reset sign after consuming a number

                if pending_op in ('*', '/'):
                    if not values:
                        raise ValueError("Malformed expression")
                    left = values.pop()
                    if pending_op == '*':
                        res = left * num
                    else:  # '/'
                        if num == 0:
                            raise ZeroDivisionError("division by zero")
                        res = left / num
                    values.append(res)
                    pending_op = None
                else:
                    values.append(num)

                expect_number = False
                i = j
                continue

            # Any other character when a number is expected is malformed
            raise ValueError("Malformed expression")
        else:
            # Expect an operator
            if ch in '*/':
                if pending_op is not None:
                    # e.g., "2**3" or "2/*3"
                    raise ValueError("Malformed expression")
                pending_op = ch
                expect_number = True
                i += 1
                continue
            if ch in '+-':
                if pending_op is not None:
                    # e.g., "2*+3" is allowed (unary + after *), but that case
                    # will only appear when expect_number is True. Here it's False.
                    raise ValueError("Malformed expression")
                # Prepare sign for the next number
                if sign != 1:
                    # Consecutive binary operators like "1+-+2" directly after a number
                    # would set sign again without consuming a number -> malformed.
                    raise ValueError("Malformed expression")
                sign = 1 if ch == '+' else -1
                expect_number = True
                i += 1
                continue

            # Any non-operator character here is malformed
            raise ValueError("Malformed expression")

    # Finished scanning
    if expect_number or pending_op is not None:
        # Expression ends with an operator or incomplete operation
        raise ValueError("Malformed expression")

    # Sum all values (addition after handling * and /)
    result = 0
    for v in values:
        result += v
    return result


def process_user_query(query: str):
    """
    Validates and evaluates a user-provided mathematical expression.
    Returns the computed result.
    Raises ValueError for unsafe or malformed input.
    """
    if not isinstance(query, str):
        raise ValueError("Expression must be a string")
    if not is_safe_query(query):
        raise ValueError("Unsupported characters in expression")
    return evaluate_expression(query)
