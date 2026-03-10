import re

def check_path(input_path: str) -> bool:
    """
    Validates whether a given string is an HTTP or HTTPS URL using regular expressions.
    
    Args:
        input_path: The URL to be validated
        
    Returns:
        True if the input is a valid HTTP or HTTPS URL format, False otherwise
    """
    # Regular expression pattern for HTTP/HTTPS URL validation
    url_pattern = r'^https?://[a-zA-Z0-9.-]+(?:\.[a-zA-Z]{2,})?(?::\d+)?(?:/[^\s]*)?$'
    
    try:
        # Check if the input matches the URL pattern
        return bool(re.match(url_pattern, input_path))
    except:
        # Return False if any exception occurs
        return False
