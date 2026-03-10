import re


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


def split_email(email):
    """
    Split a well-formed email into user and domain parts.
    
    Args:
        email (str): The email string to split
        
    Returns:
        tuple: (user, domain) if email is valid, None otherwise
    """
    if not is_valid_email(email):
        return None
    
    user, domain = email.split('@')
    return (user, domain)


def fetch_email_domain(mail_id):
    """
    Extract the domain portion from a valid email address using regex.
    
    Args:
        mail_id (str): The email string to extract domain from
        
    Returns:
        str: The domain portion if email is valid, None otherwise
    """
    try:
        if not isinstance(mail_id, str):
            return None
        
        # Regex pattern to match email and capture domain
        pattern = r'^[^@]+@([^@]+)$'
        match = re.match(pattern, mail_id)
        
        if not match:
            return None
        
        domain = match.group(1)
        
        # Validate domain has at least one dot and proper format
        if '.' not in domain or domain.startswith('.') or domain.endswith('.'):
            return None
        
        # Check there's content after the last dot
        if domain.rfind('.') == len(domain) - 1:
            return None
        
        return domain
    except:
        return None
