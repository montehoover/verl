import re

def validate_url(url):
    """
    Validates if a string is a well-formed URL.
    
    Args:
        url (str): The URL string to validate
        
    Returns:
        bool: True if valid URL format, False otherwise
    """
    # URL regex pattern
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    try:
        if not isinstance(url, str):
            return False
        return bool(url_pattern.match(url))
    except:
        return False

def extract_domain(url):
    """
    Extracts the domain from a URL if it's valid.
    
    Args:
        url (str): The URL string to extract domain from
        
    Returns:
        str: The domain part if URL is valid, False otherwise
    """
    # URL regex pattern with groups to capture domain
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'((?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)?$', re.IGNORECASE)
    
    try:
        if not isinstance(url, str):
            return False
        match = url_pattern.match(url)
        if match:
            return match.group(1)
        return False
    except:
        return False
