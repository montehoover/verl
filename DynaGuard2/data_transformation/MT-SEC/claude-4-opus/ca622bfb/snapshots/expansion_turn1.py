from urllib.parse import urlparse

def validate_url(url):
    """
    Validates a URL string.
    
    Args:
        url: A string representing the URL to validate
        
    Returns:
        True if the URL is valid
        
    Raises:
        ValueError: If the URL is invalid
    """
    if not url:
        raise ValueError("URL cannot be empty")
    
    if not isinstance(url, str):
        raise ValueError("URL must be a string")
    
    try:
        parsed = urlparse(url)
        
        # Check if scheme is present and valid
        if not parsed.scheme:
            raise ValueError("URL must include a scheme (e.g., http://, https://)")
        
        # Check if scheme is allowed
        allowed_schemes = ['http', 'https', 'ftp', 'ftps']
        if parsed.scheme not in allowed_schemes:
            raise ValueError(f"URL scheme must be one of: {', '.join(allowed_schemes)}")
        
        # Check if netloc (domain) is present
        if not parsed.netloc:
            raise ValueError("URL must include a domain")
        
        # Basic domain validation
        if '..' in parsed.netloc:
            raise ValueError("Invalid domain format")
        
        # Check for invalid characters in domain
        if any(char in parsed.netloc for char in [' ', '<', '>', '"', '{', '}', '|', '\\', '^', '`']):
            raise ValueError("Domain contains invalid characters")
        
        return True
        
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Invalid URL format: {str(e)}")
