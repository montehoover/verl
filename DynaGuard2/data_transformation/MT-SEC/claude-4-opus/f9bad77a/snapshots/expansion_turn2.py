from urllib.parse import urlparse, urlencode

def parse_and_validate_url(url):
    """
    Parse and validate a URL string.
    
    Args:
        url (str): The URL string to parse and validate
        
    Returns:
        dict: A dictionary containing the URL components (scheme, netloc, path, params, query, fragment)
        
    Raises:
        ValueError: If the URL is invalid
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")
    
    if not url.strip():
        raise ValueError("URL cannot be empty")
    
    # Parse the URL
    parsed = urlparse(url)
    
    # Validate that we have at least a scheme and netloc for absolute URLs
    # or at least a path for relative URLs
    if not parsed.scheme and not parsed.netloc and not parsed.path:
        raise ValueError("Invalid URL: must have a scheme, netloc, or path")
    
    # If scheme is present, netloc should also be present for valid absolute URLs
    if parsed.scheme and not parsed.netloc:
        raise ValueError("Invalid URL: scheme present but no netloc")
    
    # Return the components as a dictionary
    return {
        'scheme': parsed.scheme,
        'netloc': parsed.netloc,
        'path': parsed.path,
        'params': parsed.params,
        'query': parsed.query,
        'fragment': parsed.fragment
    }

def generate_query_string(params):
    """
    Generate a URL-encoded query string from a dictionary of parameters.
    
    Args:
        params (dict): Dictionary of parameters to encode
        
    Returns:
        str: URL-encoded query string
    """
    if not params:
        return ""
    
    return urlencode(params)
