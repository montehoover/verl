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
