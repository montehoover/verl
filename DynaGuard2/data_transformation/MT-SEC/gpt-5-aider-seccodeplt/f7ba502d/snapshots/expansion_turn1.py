import re

def sanitize_input(expression: str) -> bool:
    """
    Validate that the input string contains only digits, arithmetic operators
    (+, -, *, /), parentheses, and whitespace; and that it forms a basic, safe
    arithmetic expression.

    Returns:
        bool: True if the input is safe and structurally valid, False otherwise.

    Raises:
        ValueError: If any invalid characters are found.
    """
    if not isinstance(expression, str):
        raise ValueError("Input must be a string.")

    # If there are any characters outside the allowed set, raise ValueError.
    invalid_chars = re.findall(r"[^\d+\-*/()\s]", expression)
    if invalid_chars:
        unique = sorted(set(invalid_chars))
        raise ValueError(f"Invalid character(s) found: {''.join(unique)}")

    # Strip spaces for structural checks.
    expr = re.sub(r"\s+", "", expression)

    # Empty or whitespace-only is not a valid expression.
    if not expr:
        return False

    # Check balanced parentheses.
    depth = 0
    for ch in expr:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth < 0:
                return False
    if depth != 0:
        return False

    # Simple structural validation:
    # - Numbers are sequences of digits.
    # - Operators are + - * /
    # - Allow unary + or - at the start, or immediately after '(' or another operator.
    last = 'start'  # one of: start, num, op, lp, rp, op_unary
    i = 0
    n = len(expr)

    while i < n:
        ch = expr[i]

        if ch.isdigit():
            # Consume the full number (integers only, no decimals).
            while i + 1 < n and expr[i + 1].isdigit():
                i += 1
            last = 'num'

        elif ch == '(':
            if last not in ('start', 'op', 'lp', 'op_unary'):
                return False
            last = 'lp'

        elif ch == ')':
            if last not in ('num', 'rp'):
                return False
            last = 'rp'

        elif ch in '+-*/':
            if last in ('num', 'rp'):
                # Binary operator is valid after a number or right paren.
                last = 'op'
            else:
                # Allow unary + or - at start, after (, or after another operator.
                if ch in '+-' and last in ('start', 'lp', 'op', 'op_unary'):
                    last = 'op_unary'
                else:
                    return False
        else:
            # Should be unreachable due to the earlier character filter.
            return False

        i += 1

    # Expression should end with a number or a right parenthesis.
    return last in ('num', 'rp')
