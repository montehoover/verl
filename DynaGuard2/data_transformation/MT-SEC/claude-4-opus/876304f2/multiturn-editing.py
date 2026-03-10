import re

def is_valid_email(email):
    """
    Validates if an email address is correctly formatted.
    
    Args:
        email (str): The email address to validate
        
    Returns:
        bool: True if email is valid, False otherwise
    """
    # Regular expression pattern for email validation
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    # Check if the email matches the pattern
    return bool(re.match(pattern, email))

def get_email_username(email):
    """
    Extracts the username part from a valid email address.
    
    Args:
        email (str): The email address to extract username from
        
    Returns:
        str or None: The username part if email is valid, None otherwise
    """
    # Regular expression pattern for email validation with username capture group
    pattern = r'^([a-zA-Z0-9._%+-]+)@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    # Try to match the pattern
    match = re.match(pattern, email)
    
    # Return the username (first capture group) if match found, None otherwise
    return match.group(1) if match else None

def get_email_domain(addr):
    """
    Extracts the domain part from a valid email address.
    
    Args:
        addr (str): The email address to extract domain from
        
    Returns:
        str or None: The domain part if email is valid, None otherwise
    """
    # Regular expression pattern for email validation with domain capture group
    pattern = r'^[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})$'
    
    # Try to match the pattern
    match = re.match(pattern, addr)
    
    # Return the domain (first capture group) if match found, None otherwise
    return match.group(1) if match else None
