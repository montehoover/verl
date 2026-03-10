import re
from typing import Optional

def retrieve_email_domain(email_str: str) -> Optional[str]:
    """
    Validates if a given string is a properly formatted email address
    and extracts the domain part.

    Args:
        email_str: The email address string to validate.

    Returns:
        The domain part of the email if the email syntax is correct, 
        otherwise None.
    """
    # Regex for basic email validation and capturing the domain part.
    # - username part: alphanumeric characters, dots, underscores, percent, plus, hyphen
    # - @ symbol
    # - Capturing group 1: domain name part (alphanumeric characters, dots, hyphens)
    #   followed by a . (dot) and a top-level domain (at least two alphabetic characters)
    email_pattern = r"^[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})$"
    
    match = re.fullmatch(email_pattern, email_str)
    if match:
        return match.group(1)  # Return the captured domain part
    else:
        return None
