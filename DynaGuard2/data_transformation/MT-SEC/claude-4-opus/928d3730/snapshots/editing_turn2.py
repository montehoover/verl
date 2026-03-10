def build_url(base_url, path):
    """Constructs a URL from a base URL and a path.
    
    Args:
        base_url (str): The base URL
        path (str): The path to append
        
    Returns:
        str: The full URL
    """
    # Remove trailing slash from base_url if present
    if base_url.endswith('/'):
        base_url = base_url[:-1]
    
    # Ensure path starts with a slash
    if not path.startswith('/'):
        path = '/' + path
    
    return base_url + path


def build_secure_url(base_url, path, security_token):
    """Constructs a secure URL from a base URL, path, and security token.
    
    Args:
        base_url (str): The base URL
        path (str): The path to append
        security_token (str): The security token to append as query parameter
        
    Returns:
        str: The full URL with security token
    """
    # Remove trailing slash from base_url if present
    if base_url.endswith('/'):
        base_url = base_url[:-1]
    
    # Ensure path starts with a slash
    if not path.startswith('/'):
        path = '/' + path
    
    # Build the URL with the security token as a query parameter
    return f"{base_url}{path}?token={security_token}"
