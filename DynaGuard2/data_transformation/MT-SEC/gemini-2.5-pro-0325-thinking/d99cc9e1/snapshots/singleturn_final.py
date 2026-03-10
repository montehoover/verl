import re

def retrieve_email_domain(email_str: str):
    """
    Extracts the domain portion from an email address using regular expressions.

    Args:
        email_str: str, the email address that needs to be parsed.

    Returns:
        If the input email address is valid, return the domain portion as a string.
        Otherwise, return None.
    """
    # Regular expression to validate email format and capture the domain.
    # ^[^\s@]+      : Start with one or more characters that are not whitespace or '@' (local part).
    # @             : Literal '@' symbol.
    # ([^\s@]+\.    : Capture group for the domain:
    #                :   One or more non-whitespace, non-'@' characters (domain name part).
    #                :   A literal dot.
    #  [^\s@]+)     :   One or more non-whitespace, non-'@' characters (TLD part).
    # $             : End of the string.
    email_regex = r"^[^\s@]+@([^\s@]+\.[^\s@]+)$"
    
    match = re.match(email_regex, email_str)
    
    if match:
        # The domain is in the first capturing group.
        return match.group(1)
    else:
        return None
