from urllib.parse import urlparse, urljoin, quote

def validate_url(url):
    """
    Validates a URL string.
    
    Args:
        url (str): The URL to validate
        
    Returns:
        bool: True if the URL is valid
        
    Raises:
        ValueError: If the URL is invalid
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")
    
    if not url:
        raise ValueError("URL cannot be empty")
    
    try:
        result = urlparse(url)
        
        # Check if scheme and netloc are present
        if not result.scheme:
            raise ValueError("URL must include a scheme (e.g., http://, https://)")
        
        if not result.netloc:
            raise ValueError("URL must include a domain")
        
        # Check for valid schemes
        valid_schemes = ['http', 'https', 'ftp', 'ftps']
        if result.scheme.lower() not in valid_schemes:
            raise ValueError(f"Invalid URL scheme. Must be one of: {', '.join(valid_schemes)}")
        
        return True
        
    except Exception as e:
        raise ValueError(f"Invalid URL: {str(e)}")


def concatenate_url_path(base_url, path):
    """
    Concatenates a base URL with a path.
    
    Args:
        base_url (str): The base URL
        path (str): The path to append
        
    Returns:
        str: The concatenated URL
    """
    # Remove leading slash from path if present to avoid double slashes
    if path.startswith('/'):
        path = path[1:]
    
    # Ensure base_url ends with a slash
    if not base_url.endswith('/'):
        base_url = base_url + '/'
    
    # Use urljoin to properly concatenate
    result = urljoin(base_url, path)
    
    return result
