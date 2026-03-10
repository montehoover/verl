import re

def fetch_email_domain(mail_id: str):
    """
    Extracts the domain portion from an email address using regular expressions.
    
    Args:
        mail_id: str - The email address to parse
        
    Returns:
        str: The domain portion if the email is valid, None otherwise
    """
    # Regular expression pattern for validating email addresses
    # This pattern checks for basic email format: something@domain.extension
    email_pattern = r'^[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})$'
    
    # Try to match the email pattern
    match = re.match(email_pattern, mail_id)
    
    if match:
        # If valid email, return the domain (group 1 from the regex)
        return match.group(1)
    else:
        # If invalid email, return None
        return None
