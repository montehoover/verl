import re

def check_url_format(link: str) -> bool:
    """
    Verify if a given string is a valid URL based on a regular expression pattern.
    
    Args:
        link: str, the input string to validate as a URL
        
    Returns:
        True for valid URLs, and False otherwise
    """
    # Regular expression pattern for URL validation
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    if not isinstance(link, str):
        return False
    
    return bool(url_pattern.match(link))
