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
    # Regular expression to validate email and capture the domain.
    # This regex is a common one, but email validation can be complex.
    # It checks for:
    # - one or more characters (alphanumeric, dot, underscore, percent, plus, hyphen) before @
    # - an @ symbol
    # - one or more characters (alphanumeric, dot, hyphen) for the domain name part
    # - a dot
    # - two or more alphabetic characters for the top-level domain (TLD)
    email_regex = r"^[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})$"
    
    match = re.match(email_regex, email)
    
    if match:
        # The domain is in the first captured group
        return match.group(1)
    else:
        return None
