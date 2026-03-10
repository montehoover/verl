import re

def is_valid_email(email):
    """
    Validates if an email address is valid according to standard email format rules.
    
    Args:
        email (str): The email address to validate
        
    Returns:
        bool: True if the email is valid, False otherwise
    """
    # Regular expression pattern for email validation
    # This pattern checks for:
    # - One or more characters before @ (letters, numbers, dots, hyphens, underscores)
    # - @ symbol
    # - Domain name with at least one dot
    # - Top-level domain with at least 2 characters
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    try:
        # Check if email matches the pattern
        if re.match(pattern, email):
            return True
        else:
            return False
    except:
        # Return False if any exception occurs
        return False

def get_email_username(email):
    """
    Extracts the username part of an email address if valid.
    
    Args:
        email (str): The email address to extract username from
        
    Returns:
        str or None: The username part if email is valid, None otherwise
    """
    if is_valid_email(email):
        # Extract username part using regex
        match = re.match(r'^([a-zA-Z0-9._%+-]+)@', email)
        if match:
            return match.group(1)
    return None

def get_email_domain(mail_address):
    """
    Extracts the domain part of an email address if valid.
    
    Args:
        mail_address (str): The email address to extract domain from
        
    Returns:
        str or None: The domain part if email is valid, None otherwise
    """
    if is_valid_email(mail_address):
        # Extract domain part using regex
        match = re.match(r'^[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})$', mail_address)
        if match:
            return match.group(1)
    return None
