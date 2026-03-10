import re

def is_valid_path(site_path):
    """
    Determines if a given string is a valid HTTP or HTTPS path.
    
    Args:
        site_path (str): The path string to validate
        
    Returns:
        bool: True if valid path, False otherwise
    """
    # Path regex pattern for HTTP/HTTPS paths
    path_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/[^/\s]*)*/?$', re.IGNORECASE)  # path
    
    try:
        if not isinstance(site_path, str):
            return False
        return bool(path_pattern.match(site_path))
    except:
        return False
