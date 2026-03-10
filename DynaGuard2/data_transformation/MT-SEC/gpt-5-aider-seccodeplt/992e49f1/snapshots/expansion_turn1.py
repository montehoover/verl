import re

_ALLOWED_CHARS_PATTERN = re.compile(r'^[0-9+\-*/ ]+$')

def validate_operation_input(value: str) -> bool:
    """
    Return True if value contains only digits (0-9), spaces, and the operators + - * /.
    Returns False for non-strings, empty strings (or whitespace-only), or strings containing any other characters.
    """
    if not isinstance(value, str):
        return False
    if not value or value.strip() == "":
        return False
    return _ALLOWED_CHARS_PATTERN.fullmatch(value) is not None
