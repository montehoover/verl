from urllib.parse import urlparse

def parse_and_validate_url(url):
    """
    Parse and validate a URL string.
    
    Args:
        url (str): The URL string to parse and validate
        
    Returns:
        dict: A dictionary containing the URL components (scheme, domain, path)
        
    Raises:
        ValueError: If the URL is invalid
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")
    
    if not url:
        raise ValueError("URL cannot be empty")
    
    parsed = urlparse(url)
    
    # Check if scheme and netloc (domain) are present for valid URL
    if not parsed.scheme:
        raise ValueError("URL must include a scheme (e.g., http://, https://)")
    
    if not parsed.netloc:
        raise ValueError("URL must include a domain")
    
    return {
        'scheme': parsed.scheme,
        'domain': parsed.netloc,
        'path': parsed.path if parsed.path else '/'
    }
