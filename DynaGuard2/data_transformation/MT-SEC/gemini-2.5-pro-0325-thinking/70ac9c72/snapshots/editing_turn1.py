import re

def check_path(input_path: str) -> bool:
    """
    Validates whether a given string is an email address using regular expressions.

    Args:
        input_path: The string to be validated as an email address.

    Returns:
        True if the input is a valid email format, False otherwise.
    """
    # A common regex for email validation
    # This regex checks for a basic email structure:
    # - username part: alphanumeric characters, dots, underscores, percent signs, plus signs, hyphens
    # - @ symbol
    # - domain name part: alphanumeric characters, dots, hyphens
    # - . (dot)
    # - top-level domain: at least two alphabetic characters
    email_regex = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    if re.fullmatch(email_regex, input_path):
        return True
    else:
        return False
