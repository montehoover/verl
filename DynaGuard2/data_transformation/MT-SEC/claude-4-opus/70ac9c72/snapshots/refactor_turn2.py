import re

def _matches_http_pattern(text: str) -> bool:
    """
    Helper function to check if text matches the HTTP/HTTPS URL pattern.
    
    Args:
        text: str, the text to match against the pattern
        
    Returns:
        True if the text matches the pattern, otherwise False
    """
    # Regular expression pattern for valid http/https paths
    # Matches: http:// or https:// followed by domain and optional path
    pattern = r'^https?://[a-zA-Z0-9\-._~:/?#[\]@!$&\'()*+,;=%]+$'
    return bool(re.match(pattern, text))


def check_path(input_path: str) -> bool:
    """
    Determines if a given string is a valid http or https path based on matching it against a regular expression.
    
    Args:
        input_path: str, the string to be evaluated as a potential valid path
        
    Returns:
        True if the string is a valid path, otherwise False
    """
    if not isinstance(input_path, str):
        return False
        
    return _matches_http_pattern(input_path)
