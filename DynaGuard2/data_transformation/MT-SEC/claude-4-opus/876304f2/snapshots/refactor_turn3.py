import re


def get_email_domain(addr: str):
    """Extract the domain portion from an email address.
    
    Args:
        addr: The email address to examine.
        
    Returns:
        The domain part of the email if valid, None otherwise.
    """
    # Define regex pattern for valid email address
    # Pattern breakdown:
    # ^[a-zA-Z0-9._%+-]+  : Local part (before @) - alphanumeric, dots, underscores, etc.
    # @                   : Required @ symbol
    # [a-zA-Z0-9.-]+      : Domain name - alphanumeric, dots, hyphens
    # \.[a-zA-Z]{2,}$     : Top-level domain - dot followed by at least 2 letters
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    # Early return if email doesn't match valid pattern
    if not re.match(email_pattern, addr):
        return None
    
    # Email is valid, extract and return domain part after @
    return addr.split('@')[1]
