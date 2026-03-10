import re

_ALLOWED_CHARS_PATTERN = re.compile(r'^[0-9+\-*/ ]*$')

def validate_math_expression(value):
    """
    Return True if value contains only digits (0-9), '+', '-', '*', '/', and spaces.
    Otherwise return False.
    """
    if not isinstance(value, str):
        return False
    return _ALLOWED_CHARS_PATTERN.fullmatch(value) is not None
