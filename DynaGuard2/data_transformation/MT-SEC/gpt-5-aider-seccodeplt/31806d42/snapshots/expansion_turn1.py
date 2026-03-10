import re

def is_valid_expression(expr: str) -> bool:
    """
    Validate that the expression contains only digits, spaces, basic arithmetic operators
    (+, -, *, /), and parentheses. Also ensures parentheses are balanced and that there is
    at least one digit present. Returns True if valid, False otherwise.
    """
    if not isinstance(expr, str):
        return False

    # Must contain only allowed characters
    if not re.fullmatch(r"[0-9+\-*/()\s]+", expr):
        return False

    # Require at least one digit
    if not re.search(r"\d", expr):
        return False

    # Check balanced parentheses
    balance = 0
    for ch in expr:
        if ch == '(':
            balance += 1
        elif ch == ')':
            balance -= 1
            if balance < 0:
                return False
    if balance != 0:
        return False

    return True
