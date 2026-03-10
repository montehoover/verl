import re

# Precompiled regex for validating basic email address formats.
# - Local part: alphanumerics separated by single ., _, %, +, or - (no consecutive separators, no leading/trailing)
# - Domain: labels separated by dots; labels can't start or end with a hyphen
# - TLD: at least 2 letters
_EMAIL_REGEX = re.compile(
    r'^[A-Za-z0-9]+(?:[._%+-][A-Za-z0-9]+)*@(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\.)+[A-Za-z]{2,}$'
)

def check_path(input_path: str) -> bool:
    """
    Validate whether the given string is an email address.

    Args:
        input_path: The email address to be validated.

    Returns:
        True if input_path matches a valid email format, False otherwise.
    """
    if not isinstance(input_path, str):
        return False
    return _EMAIL_REGEX.fullmatch(input_path) is not None
