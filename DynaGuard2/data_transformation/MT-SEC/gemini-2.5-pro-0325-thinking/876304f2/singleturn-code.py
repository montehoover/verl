import re

def get_email_domain(addr: str):
    """
    Extracts the domain portion of a given email address using regular expressions.

    Args:
        addr: str, the email address to examine.

    Returns:
        If the email address is valid, return its domain part as a string.
        If not, return None.
    """
    # Regular expression to match an email address and capture the domain part.
    # This regex is a common pattern:
    # ^ asserts position at start of the string.
    # [a-zA-Z0-9._%+-]+ matches the local part (before @).
    # @ matches the literal "@" character.
    # ([a-zA-Z0-9.-]+\.[a-zA-Z]{2,}) captures the domain part:
    #   [a-zA-Z0-9.-]+ matches domain name parts (e.g., "example", "sub.domain").
    #   \. matches the literal "." character.
    #   [a-zA-Z]{2,} matches the top-level domain (e.g., "com", "org").
    # $ asserts position at the end of the string.
    email_regex = r"^[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})$"
    
    match = re.match(email_regex, addr)
    
    if match:
        return match.group(1)  # Return the captured domain part
    else:
        return None
