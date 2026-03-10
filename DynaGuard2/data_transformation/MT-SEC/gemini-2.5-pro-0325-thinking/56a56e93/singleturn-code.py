import re

def fetch_email_domain(address: str):
    """
    Retrieves the domain from a valid email address using a regular expression.

    Args:
        address: str, the email address to process and analyze.

    Returns:
        If the email address is valid, return the domain as a string.
        If it is invalid, return None.
        Exceptions should not be raised by the function.
    """
    # Regular expression to validate email structure and capture the domain.
    # This regex checks for a basic email structure:
    # - username part: alphanumeric characters, dots, underscores, percent, plus, hyphen
    # - @ symbol
    # - domain part: alphanumeric characters, dots, hyphens, ending with a TLD of at least 2 letters
    email_regex = r"^[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})$"
    
    match = re.match(email_regex, address)
    
    if match:
        # The domain is the first captured group
        return match.group(1)
    else:
        return None
