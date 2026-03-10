import re


def check_keyword_in_string(text, keyword):
    """
    Check if a keyword is present anywhere in the text.
    
    Args:
        text (str): The text to search in
        keyword (str): The keyword to search for
        
    Returns:
        bool: True if keyword is found in text, False otherwise
    """
    return keyword in text


def split_email_parts(email):
    """
    Split an email address into local part and domain.
    
    Args:
        email (str): The email address to split
        
    Returns:
        tuple: A tuple containing (local_part, domain)
    """
    local_part, domain = email.split('@')
    return (local_part, domain)


def retrieve_email_domain(email_str):
    """
    Extract the domain part of an email address using regular expressions.
    
    Args:
        email_str (str): The email string to parse
        
    Returns:
        str or None: The domain part if email is valid, None otherwise
    """
    # Regular expression pattern for a valid email address
    email_pattern = r'^[a-zA-Z0-9._%+-]+@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})$'
    
    match = re.match(email_pattern, email_str)
    if match:
        return match.group(1)
    return None
