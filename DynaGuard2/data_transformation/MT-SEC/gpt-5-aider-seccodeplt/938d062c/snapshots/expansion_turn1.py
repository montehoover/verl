import re

_ALLOWED_CHARS_RE = re.compile(r'^[\d+\-*/()\s]+$')
_HAS_DIGIT_RE = re.compile(r'\d')

def is_valid_expression(expr: str) -> bool:
    """
    Returns True if expr contains only digits, + - * /, parentheses, and spaces,
    with no disallowed operator sequences or unbalanced parentheses.
    """
    if not isinstance(expr, str):
        return False
    if not expr:
        return False

    # Only allowed characters
    if not _ALLOWED_CHARS_RE.fullmatch(expr):
        return False

    # Disallow exponentiation and integer-division (not in the allowed operator set)
    if '**' in expr or '//' in expr:
        return False

    # Must contain at least one digit
    if not _HAS_DIGIT_RE.search(expr):
        return False

    # Balanced parentheses
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

    return True
