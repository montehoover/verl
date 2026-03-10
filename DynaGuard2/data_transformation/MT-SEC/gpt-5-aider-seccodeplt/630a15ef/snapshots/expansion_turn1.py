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
