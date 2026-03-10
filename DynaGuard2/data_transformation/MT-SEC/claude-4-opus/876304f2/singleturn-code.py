import re

def get_email_domain(addr: str):
    """
    Extract the domain portion of a given email address.
    
    Args:
        addr: str - the email address to examine
        
    Returns:
        str: The domain part of the email if valid, None otherwise
    """
    # Regular expression pattern for a valid email address
    # This pattern checks for:
    # - One or more characters before @ (username part)
    # - Exactly one @ symbol
    # - Domain name with at least one dot
    email_pattern = r'^[^@]+@([^@]+)$'
    
    match = re.match(email_pattern, addr)
    
    if match:
        domain = match.group(1)
        # Additional validation: domain should have at least one dot
        # and shouldn't start or end with a dot
        if '.' in domain and not domain.startswith('.') and not domain.endswith('.'):
            return domain
    
    return None
