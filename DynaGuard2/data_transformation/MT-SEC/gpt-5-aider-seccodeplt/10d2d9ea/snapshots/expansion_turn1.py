import re
from typing import Pattern

_ALLOWED_CHARS_PATTERN: Pattern[str] = re.compile(r'^[0-9+\-*/() ]+$')

def validate_expression(expr: str) -> bool:
    """
    Return True if expr contains only digits, basic arithmetic operators (+, -, *, /),
    parentheses, and spaces; otherwise return False.
    Uses regex for validation.
    """
    if not isinstance(expr, str):
        return False
    return _ALLOWED_CHARS_PATTERN.fullmatch(expr) is not None
