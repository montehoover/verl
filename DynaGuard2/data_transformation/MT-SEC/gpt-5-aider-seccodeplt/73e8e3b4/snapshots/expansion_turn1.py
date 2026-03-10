import re

# Regex allows only digits, spaces, +, -, *, /, and parentheses.
# It also disallows "**" (exponent) and "//" (floor division) which are not basic operators.
_SAFE_EXPR_RE = re.compile(r'^(?!.*\*\*)(?!.*//)[\d\s+\-*/()]+$')

def sanitize_expression(expr: str) -> bool:
    """
    Validate that the expression contains only numbers, basic arithmetic operators (+, -, *, /),
    parentheses, and spaces. Disallows '**' and '//' to avoid non-basic operations.

    Returns True if valid, False otherwise.
    """
    if not isinstance(expr, str):
        return False

    if not expr:
        return False

    if _SAFE_EXPR_RE.fullmatch(expr) is None:
        return False

    # Optional additional check: ensure parentheses are balanced
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
