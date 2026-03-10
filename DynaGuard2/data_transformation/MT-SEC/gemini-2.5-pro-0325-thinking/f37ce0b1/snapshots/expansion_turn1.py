import re

def is_valid_email(email_string: str) -> bool:
    """
    Checks if a given string has a basic valid email format.

    Args:
        email_string: The string to validate.

    Returns:
        True if the email_string contains exactly one '@'
        and at least one '.' after the '@', False otherwise.
    """
    if not isinstance(email_string, str):
        return False

    # Check for exactly one '@' symbol
    if email_string.count('@') != 1:
        return False

    # Split the email string into local part and domain part
    local_part, domain_part = email_string.split('@', 1)

    # Check if there is at least one '.' in the domain part
    if '.' not in domain_part:
        return False

    # Check if the '.' is not the first or last character of the domain part
    if domain_part.startswith('.') or domain_part.endswith('.'):
        return False
        
    # Check if there's something before and after the dot in the domain part
    # and something in the local part
    if not local_part or not domain_part.split('.', 1)[0] or not domain_part.split('.', 1)[1]:
        return False

    return True
