import re

def is_valid_email(email):
    """
    Check if a given string is a valid email address.
    
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
    else:
        return False

def split_email_components(email):
    """
    Split an email address into its username and domain components.
    
    Args:
        email (str): The email address to split
        
    Returns:
        dict: A dictionary with 'username' and 'domain' keys
    """
    if '@' in email:
        username, domain = email.split('@', 1)
        return {
            'username': username,
            'domain': domain
        }
    else:
        return {
            'username': '',
            'domain': ''
        }

def fetch_email_domain(address):
    """
    Extract the domain from a valid email address using regex.
    
    Args:
        address (str): The email address to parse
        
    Returns:
        str or None: The domain if the email is valid, None otherwise
    """
    # Regular expression pattern to capture email components
    pattern = r'^[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})$'
    
    # Try to match the pattern
    match = re.match(pattern, address)
    
    if match:
        return match.group(1)
    else:
        return None
