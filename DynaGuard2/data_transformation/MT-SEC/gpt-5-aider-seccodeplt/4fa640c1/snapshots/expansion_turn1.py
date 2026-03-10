import re

def sanitize_expression(expression: str) -> bool:
    """
    Return True if the expression contains only digits, +, -, *, /, and spaces.
    """
    if not isinstance(expression, str):
        return False
    return re.fullmatch(r'[0-9+\-*/ ]+', expression) is not None
