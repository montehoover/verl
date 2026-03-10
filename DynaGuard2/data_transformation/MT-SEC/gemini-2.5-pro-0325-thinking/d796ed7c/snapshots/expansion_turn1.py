import re

def sanitize_input(input_string: str) -> bool:
    """
    Checks if a given string only contains digits, basic arithmetic operators
    (+, -, *, /), and spaces.

    Args:
        input_string: The string to sanitize.

    Returns:
        True if the input string is valid, False otherwise.
    """
    allowed_chars_pattern = r"^[0-9+\-*/\s]*$"
    if re.fullmatch(allowed_chars_pattern, input_string):
        return True
    return False
