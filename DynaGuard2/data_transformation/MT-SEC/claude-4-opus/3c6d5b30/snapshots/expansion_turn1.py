from urllib.parse import urlparse

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
