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
    # Regular expression to match a valid email address and capture the domain part.
    # The domain is the part after '@' and should conform to typical domain naming rules.
    # This regex checks for:
    #   - local part: one or more alphanumeric characters, dots, underscores, percent signs, plus or minus signs
    #   - '@' symbol
    #   - domain part: one or more alphanumeric characters or hyphens, followed by a dot,
    #     and then at least two alphabetic characters for the TLD.
    # The domain part is captured in a group.
    email_regex = r"^[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})$"
    
    match = re.fullmatch(email_regex, address)
    
    if match:
        # If the entire string matches the email pattern,
        # group(1) will return the captured domain.
        return match.group(1)
    else:
        # If the email structure is not valid, return None.
        return None
