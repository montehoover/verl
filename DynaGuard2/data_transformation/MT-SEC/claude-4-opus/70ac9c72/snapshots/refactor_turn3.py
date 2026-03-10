import re


def _matches_http_pattern(text: str) -> bool:
    """
    Helper function to check if text matches the HTTP/HTTPS URL pattern.
    
    This function uses a regular expression to validate that the provided
    text conforms to the structure of an HTTP or HTTPS URL.
    
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
    Determines if a given string is a valid http or https path.
    
    This function validates whether the input string represents a valid
    HTTP or HTTPS URL by checking its type and matching it against a
    regular expression pattern.
    
    Args:
        input_path: str, the string to be evaluated as a potential valid path
        
    Returns:
        True if the string is a valid path, otherwise False
        
    Examples:
        >>> check_path("https://example.com")
        True
        >>> check_path("http://example.com/path")
        True
        >>> check_path("ftp://example.com")
        False
        >>> check_path(123)
        False
    """
    if not isinstance(input_path, str):
        return False
    
    return _matches_http_pattern(input_path)
