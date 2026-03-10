import re

def retrieve_email_domain(email_str: str) -> bool:
    """
    Validates if a given string is a properly formatted email address.

    Args:
        email_str: The email address string to validate.

    Returns:
        True if the email syntax is correct, False otherwise.
    """
    # Regex for basic email validation
    # This regex checks for a common pattern:
    # - username part: alphanumeric characters, dots, underscores, percent, plus, hyphen
    # - @ symbol
    # - domain name part: alphanumeric characters, dots, hyphens
    # - . (dot)
    # - top-level domain: at least two alphabetic characters
    email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    
    if re.fullmatch(email_pattern, email_str):
        return True
    else:
        return False
