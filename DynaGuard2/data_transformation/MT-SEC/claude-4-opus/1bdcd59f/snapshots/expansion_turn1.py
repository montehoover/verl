def is_http_protocol(url):
    """
    Check if a string starts with 'http' or 'https'.
    
    Args:
        url (str): The string to check
        
    Returns:
        bool: True if the string starts with 'http' or 'https', False otherwise
    """
    if not isinstance(url, str):
        return False
    
    return url.startswith('http://') or url.startswith('https://')
