from urllib.parse import urlparse, urlencode

def parse_and_validate_url(url):
    """
    Parse and validate a URL string.
    
    Args:
        url (str): The URL string to parse and validate
        
    Returns:
        dict: A dictionary containing the URL components (scheme, netloc, path, params, query, fragment)
        
    Raises:
        ValueError: If the URL is invalid or missing required components
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")
    
    if not url.strip():
        raise ValueError("URL cannot be empty")
    
    # Parse the URL
    parsed = urlparse(url)
    
    # Validate that the URL has at least a scheme and netloc
    if not parsed.scheme:
        raise ValueError("URL must have a valid scheme (e.g., http, https)")
    
    if not parsed.netloc:
        raise ValueError("URL must have a valid network location (domain)")
    
    # Return the parsed components as a dictionary
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
        params (dict): A dictionary of query parameters
        
    Returns:
        str: A URL-encoded query string
    """
    if not params:
        return ""
    
    return urlencode(params)
