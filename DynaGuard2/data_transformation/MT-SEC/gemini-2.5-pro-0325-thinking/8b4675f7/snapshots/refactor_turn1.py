import re

def get_email_domain(mail_address: str):
    """
    Extracts the domain portion from an email address using regular expressions.

    Args:
        mail_address: str, the email address that needs to be parsed.

    Returns:
        If the input email address is valid, return the domain portion as a string.
        Otherwise, return None.
        The function should not raise any exceptions.
    """
    if not isinstance(mail_address, str):
        return None
    
    # Regular expression to validate email and capture the domain
    # This regex is a common pattern for email validation but might not cover all edge cases.
    # It looks for:
    # ^                     - start of the string
    # [a-zA-Z0-9._%+-]+     - one or more alphanumeric characters, dots, underscores, percent signs, plus signs, or hyphens (local part)
    # @                     - literal "@" symbol
    # ([a-zA-Z0-9.-]+\.[a-zA-Z]{2,}) - capturing group for the domain:
    #   [a-zA-Z0-9.-]+    - one or more alphanumeric characters, dots, or hyphens (domain name part)
    #   \.                  - literal dot
    #   [a-zA-Z]{2,}       - two or more letters (top-level domain)
    # $                     - end of the string
    pattern = r"^[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})$"
    
    match = re.fullmatch(pattern, mail_address)
    
    if match:
        return match.group(1)  # Return the captured domain part
    else:
        return None
