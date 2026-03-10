import re

def retrieve_domain_from_email(email_address: str):
    """
    Retrieves the domain from a valid email address using a regular expression.

    Args:
        email_address: str, the email address to process and analyze.

    Returns:
        If the email address is valid, return the domain as a string.
        If it is invalid, return None.
        Exceptions should not be raised by the function.
    """
    # Regex to validate email structure and capture the domain.
    # The domain is the part after '@' and should look like 'example.com'.
    # Regex to validate email structure and capture the domain.
    # The domain is the part after '@' and should look like 'example.com'.
    # PEP-8: Using a more descriptive name for the regex pattern, though email_regex was also acceptable.
    email_pattern = r"^[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})$"
    
    match = re.match(email_pattern, email_address)
    
    # Guard clause: if the email address doesn't match the pattern, return None immediately.
    if not match:
        return None
    
    # If a match is found, the first capturing group is the domain.
    return match.group(1)
