import re

def extract_email_domain(email):
    """
    Extract the domain part from a valid email address.
    
    Args:
        email (str): The email address to validate and extract from
        
    Returns:
        str or None: The domain part if the email is valid, None otherwise
    """
    # Regular expression pattern for email validation and domain extraction
    pattern = r'^[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})$'
    
    # Check if the email matches the pattern
    match = re.match(pattern, email)
    if match:
        return match.group(1)
    else:
        return None
