from urllib.parse import urlparse

def validate_and_parse_url(url):
    """
    Validates and parses a URL string.
    
    Args:
        url (str): The URL string to validate and parse
        
    Returns:
        dict: A dictionary containing the URL components (scheme, domain, path)
        
    Raises:
        ValueError: If the URL is invalid
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")
    
    if not url.strip():
        raise ValueError("URL cannot be empty")
    
    parsed = urlparse(url)
    
    # Check if scheme is present and valid
    if not parsed.scheme:
        raise ValueError("URL must include a scheme (e.g., http:// or https://)")
    
    # Check if domain/netloc is present
    if not parsed.netloc:
        raise ValueError("URL must include a domain")
    
    return {
        'scheme': parsed.scheme,
        'domain': parsed.netloc,
        'path': parsed.path if parsed.path else '/'
    }
