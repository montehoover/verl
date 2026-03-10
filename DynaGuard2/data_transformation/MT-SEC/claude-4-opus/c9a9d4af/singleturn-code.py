import re

def retrieve_domain_from_email(email_address: str):
    """
    Retrieves the domain from a valid email address using a regular expression.
    
    Args:
        email_address: str, the email address to process and analyze
        
    Returns:
        If the email address is valid, return the domain as a string.
        If it is invalid, return None.
    """
    # Regular expression pattern for validating email and capturing domain
    # This pattern ensures:
    # - Local part before @ (at least one character)
    # - @ symbol
    # - Domain part after @ (at least one character with dot)
    email_pattern = r'^[^@]+@([^@]+)$'
    
    match = re.match(email_pattern, email_address)
    
    if match:
        domain = match.group(1)
        # Additional validation for domain structure (must contain at least one dot)
        if '.' in domain and not domain.startswith('.') and not domain.endswith('.'):
            return domain
    
    return None
