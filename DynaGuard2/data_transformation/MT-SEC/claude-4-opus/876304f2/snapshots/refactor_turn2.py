import re


def get_email_domain(addr: str):
    """Extract the domain portion from an email address.
    
    Args:
        addr: The email address to examine.
        
    Returns:
        The domain part of the email if valid, None otherwise.
    """
    # Define regex pattern for valid email address
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    # Validate email format using regex
    if re.match(email_pattern, addr):
        # Extract domain (everything after @)
        return addr.split('@')[1]
    
    return None
