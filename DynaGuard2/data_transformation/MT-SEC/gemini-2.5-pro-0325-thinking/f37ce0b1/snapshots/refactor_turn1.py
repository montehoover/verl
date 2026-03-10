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
    # Regular expression to validate email and capture the domain part.
    # This regex checks for a basic email structure:
    # - local part: alphanumeric characters, dots, underscores, percent, plus, hyphen
    # - @ symbol
    # - domain part: alphanumeric characters, dots, hyphens, ending with a TLD of at least 2 letters.
    # The domain part is captured in a group.
    email_regex = r"^[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})$"
    
    match = re.match(email_regex, mail_id)
    
    if match:
        return match.group(1)  # Return the captured domain
    else:
        return None
