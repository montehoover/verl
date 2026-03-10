import re

def is_valid_expression(expr: str) -> bool:
    if not isinstance(expr, str) or not expr:
        return False
    # Allow only digits, +, -, *, /, parentheses, and spaces
    if not re.fullmatch(r'[0-9+\-*/() ]+', expr):
        return False
    # Ensure there's at least one digit
    if not re.search(r'\d', expr):
        return False
    return True
