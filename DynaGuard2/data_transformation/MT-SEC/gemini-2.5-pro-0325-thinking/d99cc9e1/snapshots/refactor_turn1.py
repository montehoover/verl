import re

def retrieve_email_domain(email_str: str):
    """
    Extracts the domain portion from an email address using regular expressions.

    Args:
        email_str: str, the email address that needs to be parsed.

    Returns:
        If the input email address is valid, return the domain portion as a string.
        Otherwise, return None.
    
    Raises:
        The function should not raise any exceptions.
    """
    if not isinstance(email_str, str):
        return None
    
    # Regular expression to match a valid email address and capture the domain part.
    # This regex is a common one, but email validation can be complex.
    # It looks for one or more characters (not '@') before the '@' symbol,
    # then the '@' symbol, then the domain part.
    # The domain part consists of one or more characters (alphanumeric, hyphen)
    # followed by a dot, and then two or more alphabetic characters (e.g., .com, .org).
    # This regex is a simplified version for common email formats.
    match = re.fullmatch(r"[^@]+@([^@]+\.[a-zA-Z]{2,})", email_str)
    
    if match:
        # The domain is in the first capturing group
        return match.group(1)
    else:
        return None
