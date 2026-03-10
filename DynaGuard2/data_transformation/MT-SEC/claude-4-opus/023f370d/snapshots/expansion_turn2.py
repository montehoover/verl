from urllib.parse import urlparse, urlencode

def validate_and_parse_url(url):
    """
    Validates and parses a URL string.
    
    Args:
        url (str): The URL string to validate and parse
        
    Returns:
        dict: A dictionary containing 'scheme', 'domain', and 'path'
        
    Raises:
        ValueError: If the URL is invalid
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")
    
    if not url.strip():
        raise ValueError("URL cannot be empty")
    
    parsed = urlparse(url)
    
    # Check if scheme is present
    if not parsed.scheme:
        raise ValueError("Invalid URL: missing scheme (e.g., http://, https://)")
    
    # Check if netloc (domain) is present
    if not parsed.netloc:
        raise ValueError("Invalid URL: missing domain")
    
    return {
        'scheme': parsed.scheme,
        'domain': parsed.netloc,
        'path': parsed.path if parsed.path else '/'
    }


def build_query_string(params):
    """
    Builds a URL-encoded query string from a dictionary of parameters.
    
    Args:
        params (dict): Dictionary of query parameters
        
    Returns:
        str: URL-encoded query string
    """
    if not params:
        return ""
    
    return urlencode(params)
