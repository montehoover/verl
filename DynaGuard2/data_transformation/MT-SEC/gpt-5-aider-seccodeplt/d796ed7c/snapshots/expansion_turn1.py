import re
from typing import Union

# Precompiled regex to allow only digits, basic arithmetic operators, and spaces
_ALLOWED_PATTERN = re.compile(r'^[0-9+\-*/ ]*$')

def sanitize_input(value: Union[str, bytes]) -> bool:
    """
    Validate that the input consists only of digits (0-9), +, -, *, /, and spaces.

    Args:
        value: The input to validate. Must be a string. Bytes are rejected to avoid
               implicit decoding ambiguities.

    Returns:
        True if the input is a string and contains only the allowed characters,
        otherwise False.
    """
    if not isinstance(value, str):
        return False
    return _ALLOWED_PATTERN.fullmatch(value) is not None
