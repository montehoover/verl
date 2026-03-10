import re

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


def validate_url_structure(url):
    """
    Check if a given string contains a valid domain and path.
    
    Args:
        url (str): The URL string to validate
        
    Returns:
        bool: True if the string contains a valid domain and path, False otherwise
    """
    if not isinstance(url, str):
        return False
    
    # Regular expression pattern for URL validation
    # This pattern checks for:
    # - Optional protocol (http:// or https://)
    # - Domain name with subdomains
    # - Optional port number
    # - Optional path
    # - Optional query parameters
    # - Optional fragment
    url_pattern = re.compile(
        r'^(https?://)?' # Optional protocol
        r'(([a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)*' # Subdomains
        r'[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)' # Domain name
        r'(:[0-9]{1,5})?' # Optional port
        r'(/[a-zA-Z0-9\-._~:/?#[\]@!$&\'()*+,;=%]*)?' # Optional path and query
        r'$'
    )
    
    return bool(url_pattern.match(url))


def path_check(u):
    """
    Check if a string is a valid http or https path using regex.
    
    Args:
        u (str): The string to check
        
    Returns:
        bool: True if the string matches the path format, False otherwise
    """
    try:
        if not isinstance(u, str):
            return False
        
        # Regex pattern for validating full HTTP/HTTPS URLs
        # Pattern checks for:
        # - Required protocol (http:// or https://)
        # - Domain with optional subdomains
        # - Optional port
        # - Optional path, query, and fragment
        path_pattern = re.compile(
            r'^https?://'  # Required http:// or https://
            r'(([a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)*'  # Optional subdomains
            r'[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?)'  # Domain
            r'(:[0-9]{1,5})?'  # Optional port
            r'(/[a-zA-Z0-9\-._~:/?#[\]@!$&\'()*+,;=%]*)?'  # Optional path
            r'$'
        )
        
        return bool(path_pattern.match(u))
    except:
        return False
