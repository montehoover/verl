import re

def is_valid_email_format(email):
    """
    Check if a given string is a valid email format.
    
    Args:
        email (str): The email address to validate
        
    Returns:
        bool: True if the email is in a valid format, False otherwise
    """
    # Regular expression pattern for email validation
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    # Check if the email matches the pattern
    return bool(re.match(pattern, email))

def get_email_username(email):
    """
    Extract the username part from an email address.
    
    Args:
        email (str): The email address to extract username from
        
    Returns:
        str or None: The username part of the email, or None if invalid
    """
    if not is_valid_email_format(email):
        return None
    
    return email.split('@')[0]

def extract_email_domain(email):
    """
    Extract the domain part from an email address using regex.
    
    Args:
        email (str): The email address to extract domain from
        
    Returns:
        str or None: The domain part of the email, or None if invalid
    """
    try:
        # Pattern to match email and capture domain part
        pattern = r'^[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})$'
        match = re.match(pattern, email)
        
        if match:
            return match.group(1)
        else:
            return None
    except:
        return None
