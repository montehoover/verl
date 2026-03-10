import re

def get_email_domain(mail_address: str):
    """
    Extracts the domain portion from an email address using regular expressions.

    Args:
        mail_address: str, the email address that needs to be parsed.

    Returns:
        If the input email address is valid, return the domain portion as a string.
        Otherwise, return None.
    """
    # Regular expression for validating an email and capturing the domain.
    # It checks for a common email format:
    # - Local part: one or more alphanumeric characters, dots, underscores, percent signs, plus or minus.
    # - @ symbol.
    # - Domain part: one or more alphanumeric characters or hyphens, followed by a dot,
    #   and then at least two alphabetic characters (the TLD).
    pattern = r"^[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})$"
    
    match = re.match(pattern, mail_address)
    
    if match:
        # The first captured group (index 1) is the domain.
        return match.group(1)
    else:
        return None
