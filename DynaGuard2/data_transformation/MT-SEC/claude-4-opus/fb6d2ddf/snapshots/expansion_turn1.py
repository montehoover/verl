from urllib.parse import urlparse

def parse_and_validate_url(url):
    """
    Parse and validate a URL string.
    
    Args:
        url (str): The URL string to parse and validate
        
    Returns:
        tuple: A tuple containing (scheme, netloc, path) components of the URL
        
    Raises:
        ValueError: If the URL is not valid
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")
    
    if not url.strip():
        raise ValueError("URL cannot be empty")
    
    parsed = urlparse(url)
    
    # Check if the URL has at least a scheme and netloc
    if not parsed.scheme:
        raise ValueError("URL must have a valid scheme (e.g., http, https)")
    
    if not parsed.netloc:
        raise ValueError("URL must have a valid network location")
    
    return (parsed.scheme, parsed.netloc, parsed.path)
