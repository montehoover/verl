import re

def fetch_email_domain(mail_id: str):
    """
    Extract the domain portion from an email address.
    
    Args:
        mail_id (str): The email address to extract from
        
    Returns:
        str: The domain part (e.g., "example.com") or None if invalid
    """
    # Regular expression to match email address and capture domain
    pattern = r'^[^@]+@([^@]+)$'
    
    match = re.match(pattern, mail_id)
    if match:
        return match.group(1)
    return None
