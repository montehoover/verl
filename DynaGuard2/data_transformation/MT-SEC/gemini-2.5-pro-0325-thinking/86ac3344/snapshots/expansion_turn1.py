import re

def validate_math_expression(expression: str) -> bool:
    """
    Validates a mathematical expression string.

    Args:
        expression: The string to validate.

    Returns:
        True if the string only contains numbers, arithmetic operators
        (+, -, *, /), and spaces, False otherwise.
    """
    allowed_chars_pattern = r"^[0-9+\-*/\s]*$"
    return bool(re.fullmatch(allowed_chars_pattern, expression))
