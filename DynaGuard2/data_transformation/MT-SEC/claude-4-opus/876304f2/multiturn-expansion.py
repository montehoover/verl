import re

def is_valid_email(email):
    """
    Validates if a given string is a valid email address.
    
    Args:
        email (str): The email address to validate
        
    Returns:
        bool: True if the email is valid, False otherwise
    """
    # Regular expression pattern for email validation
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    # Check if the email matches the pattern
    if re.match(pattern, email):
        return True
    return False

def split_email(email):
    """
    Splits a valid email address into username and domain parts.
    
    Args:
        email (str): The email address to split
        
    Returns:
        tuple: A tuple containing (username, domain)
    """
    username, domain = email.split('@')
    return (username, domain)

def get_email_domain(addr):
    """
    Extracts the domain part of an email address using regular expressions.
    
    Args:
        addr (str): The email address to extract domain from
        
    Returns:
        str or None: The domain part if email is valid, None otherwise
    """
    try:
        # Pattern to match email and capture the domain part
        pattern = r'^[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})$'
        match = re.match(pattern, addr)
        
        if match:
            return match.group(1)
        return None
    except:
        return None
