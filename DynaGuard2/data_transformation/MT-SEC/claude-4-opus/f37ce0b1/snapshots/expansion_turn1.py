def is_valid_email(email):
    """
    Check if a string is a valid email format.
    
    Args:
        email (str): The email string to validate
        
    Returns:
        bool: True if valid email format, False otherwise
    """
    if not isinstance(email, str):
        return False
    
    # Check for exactly one '@' symbol
    if email.count('@') != 1:
        return False
    
    # Split by '@' to get local and domain parts
    local, domain = email.split('@')
    
    # Check that both parts exist
    if not local or not domain:
        return False
    
    # Check for at least one '.' in the domain part
    if '.' not in domain:
        return False
    
    # Check that '.' is not at the start or end of domain
    if domain.startswith('.') or domain.endswith('.'):
        return False
    
    # Check that there's content after the last '.'
    last_dot_index = domain.rfind('.')
    if last_dot_index == len(domain) - 1:
        return False
    
    return True
