from urllib.parse import urlparse

def parse_and_validate_url(url):
    """
    Parse and validate a URL string.
    
    Args:
        url (str): The URL string to parse and validate
        
    Returns:
        tuple: A tuple containing (scheme, netloc, path)
        
    Raises:
        ValueError: If the URL is invalid or missing required components
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")
    
    if not url.strip():
        raise ValueError("URL cannot be empty")
    
    parsed = urlparse(url)
    
    # Check if scheme and netloc are present (minimum requirements for a valid URL)
    if not parsed.scheme:
        raise ValueError("URL must include a scheme (e.g., http, https)")
    
    if not parsed.netloc:
        raise ValueError("URL must include a network location (e.g., domain name)")
    
    # Return the required components
    return (parsed.scheme, parsed.netloc, parsed.path)
