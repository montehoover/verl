import re
from typing import Optional

def retrieve_email_domain(email_str: str) -> Optional[str]:
    """
    Validates if a given string is a properly formatted email address
    and extracts the username part.

    Args:
        email_str: The email address string to validate.

    Returns:
        The username part of the email if the email syntax is correct, 
        otherwise None.
    """
    # Regex for basic email validation and capturing the username part.
    # - Capturing group 1: username part (alphanumeric characters, dots, underscores, percent, plus, hyphen)
    # - @ symbol
    # - domain name part: alphanumeric characters, dots, hyphens
    # - . (dot)
    # - top-level domain: at least two alphabetic characters
    email_pattern = r"^([a-zA-Z0-9._%+-]+)@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    
    match = re.fullmatch(email_pattern, email_str)
    if match:
        return match.group(1)  # Return the captured username part
    else:
        return None
