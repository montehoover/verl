import re

def is_valid_email(email_string: str) -> bool:
    """
    Checks if a given string has a basic valid email format.

    Args:
        email_string: The string to validate.

    Returns:
        True if the email_string contains exactly one '@'
        and at least one '.' after the '@', False otherwise.
    """
    if not isinstance(email_string, str):
        return False

    # Check for exactly one '@' symbol
    if email_string.count('@') != 1:
        return False

    # Split the email string into local part and domain part
    local_part, domain_part = email_string.split('@', 1)

    # Check if there is at least one '.' in the domain part
    if '.' not in domain_part:
        return False

    # Check if the '.' is not the first or last character of the domain part
    if domain_part.startswith('.') or domain_part.endswith('.'):
        return False
        
    # Check if there's something before and after the dot in the domain part
    # and something in the local part
    if not local_part or not domain_part.split('.', 1)[0] or not domain_part.split('.', 1)[1]:
        return False

    return True

def split_email(email_string: str) -> tuple[str, str] | None:
    """
    Splits a well-formed email string into its user and domain parts.

    Args:
        email_string: The email string to split.

    Returns:
        A tuple containing the user part and the domain part if the email
        is valid, otherwise None.
    """
    if not is_valid_email(email_string):
        return None
    
    user_part, domain_part = email_string.split('@', 1)
    return user_part, domain_part

def fetch_email_domain(mail_id: str) -> str | None:
    """
    Extracts the domain part from a valid email address using regular expressions.

    Args:
        mail_id: The email string to extract the domain from.

    Returns:
        The domain as a string if the email is properly formatted, 
        otherwise None.
    """
    if not is_valid_email(mail_id):
        return None
    
    # is_valid_email ensures there is exactly one '@' and content after it.
    # So, this regex search should reliably find the domain part.
    match = re.search(r'@(.+)', mail_id)
    if match:
        return match.group(1)
    else:
        # This case should ideally not be reached if is_valid_email is robust
        return None
