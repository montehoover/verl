import re

def is_valid_string(s):
    """
    Check if a string is non-empty and doesn't contain any spaces.
    
    Args:
        s: The string to validate
        
    Returns:
        bool: True if the string is non-empty and contains no spaces, False otherwise
    """
    return len(s) > 0 and ' ' not in s

def check_url_format(link):
    """
    Check if a string is a valid URL using regular expression.
    
    Args:
        link: The string to check for valid URL format
        
    Returns:
        bool: True if the URL is formatted correctly, False otherwise
    """
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    try:
        return bool(url_pattern.match(link))
    except:
        return False
