import re

def fetch_email_domain(mail_id: str):
    """
    Extracts the domain portion from an email address using regular expressions.

    Args:
        mail_id: str, the email address that needs to be parsed.

    Returns:
        If the input email address is valid, return the domain portion as a string.
        Otherwise, return None.
        The function should not raise any exceptions.
    """
    # Regex to validate email and capture domain
    # It checks for a basic structure: local-part@domain-part
    # Domain part consists of one or more labels separated by dots, ending with a TLD of at least 2 letters.
    email_regex = r"^[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})$"
    
    match = re.match(email_regex, mail_id)
    
    if match:
        # The domain is the first captured group
        return match.group(1)
    else:
        return None
