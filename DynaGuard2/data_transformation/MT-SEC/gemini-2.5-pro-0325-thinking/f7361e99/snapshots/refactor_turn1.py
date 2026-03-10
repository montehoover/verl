import re

def extract_email_domain(email: str):
    """
    Extracts the domain part from an email address using a regular expression.

    Args:
        email: str, the input email address to be processed.

    Returns:
        If the input is a valid email address, return the domain part as a string.
        Otherwise, return None.
    """
    # A common regex for email validation, though perfect email regex is complex.
    # This regex checks for a basic structure: something@something.something
    # and captures the domain part.
    match = re.fullmatch(r"[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", email)
    if match:
        return match.group(1)  # group(1) is the captured domain
    return None
