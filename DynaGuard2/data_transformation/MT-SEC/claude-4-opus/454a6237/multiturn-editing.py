import re

def check_path_validity(path_string: str) -> bool:
    """
    Verify if a given string is a valid HTTP or HTTPS URL using regular expressions.
    
    Args:
        path_string: The string to validate as an HTTP/HTTPS URL
        
    Returns:
        True if the string is a valid HTTP/HTTPS URL, False otherwise
    """
    # Regular expression pattern for HTTP/HTTPS URL validation
    url_pattern = r'^https?://[a-zA-Z0-9.-]+(\.[a-zA-Z]{2,})?(/.*)?$'
    
    # Check if the string matches the URL pattern
    try:
        return bool(re.match(url_pattern, path_string))
    except:
        return False
