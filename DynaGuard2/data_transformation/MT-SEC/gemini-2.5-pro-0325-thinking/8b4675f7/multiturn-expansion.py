import re
from typing import Optional

def is_valid_email(email: str) -> bool:
    """
    Checks if a given string is a valid email address.

    Args:
        email: The string to validate.

    Returns:
        True if the email is valid, False otherwise.
    """
    # A common regex for email validation
    # This regex is not exhaustive but covers most common cases.
    # For a more robust solution, a dedicated library might be better.
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    if re.match(pattern, email):
        return True
    return False

def split_email_components(email: str) -> dict:
    """
    Splits an email address into its username and domain components.

    Args:
        email: The email address string.

    Returns:
        A dictionary with 'username' and 'domain' keys.
        Returns {'username': None, 'domain': None} if '@' is not found.
    """
    parts = email.split('@', 1)
    if len(parts) == 2:
        return {'username': parts[0], 'domain': parts[1]}
    else:
        # Handle cases where '@' is not found or email format is unexpected
        return {'username': None, 'domain': None}

def get_email_domain(mail_address: str) -> Optional[str]:
    """
    Extracts the domain portion from an email address using regular expressions.

    Args:
        mail_address: The email address string.

    Returns:
        The domain portion as a string if the input email address is valid
        and a domain is found, or None otherwise.
    """
    # Regex to validate email structure and capture the domain part.
    # The domain part is in the first capturing group.
    pattern = r"^[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})$"
    match = re.match(pattern, mail_address)
    if match:
        return match.group(1)  # Return the captured domain
    return None
