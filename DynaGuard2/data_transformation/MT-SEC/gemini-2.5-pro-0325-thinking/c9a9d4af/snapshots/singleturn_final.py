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
    # Regular expression to validate email structure and capture the domain.
    # This regex checks for a basic email structure:
    # - local-part@domain-part
    # - local-part can contain alphanumeric characters, ., _, %, +, -
    # - domain-part consists of labels separated by dots, ending with a TLD of at least 2 letters.
    # - Each label in the domain can contain alphanumeric characters and hyphens.
    # - Hyphens are not allowed at the beginning or end of domain labels.
    email_regex = r"^[a-zA-Z0-9._%+-]+@([a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)+)$"
    
    match = re.fullmatch(email_regex, email_address)
    
    if match:
        return match.group(1)  # group(1) is the captured domain
    else:
        return None
