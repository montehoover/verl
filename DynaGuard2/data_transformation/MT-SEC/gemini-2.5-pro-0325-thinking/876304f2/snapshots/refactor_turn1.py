import re

def get_email_domain(addr: str):
    """
    Extracts the domain portion of a given email address using regular expressions.

    Args:
        addr: str, the email address to examine.

    Returns:
        If the email address is correctly structured, returns its domain part as a string.
        If not, returns None.
    """
    # Regex to validate the email structure and capture the domain.
    # ^[a-zA-Z0-9._%+-]+      : Matches the local part of the email.
    # @                       : Matches the literal '@' symbol.
    # ([a-zA-Z0-9.-]+\.      : Captures the domain name part (e.g., "example").
    #  [a-zA-Z]{2,})         : Captures the top-level domain (e.g., "com").
    # $                       : Matches the end of the string.
    pattern = r"^[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})$"
    
    match = re.match(pattern, addr)
    
    if match:
        return match.group(1)  # The first captured group is the domain.
    else:
        return None
