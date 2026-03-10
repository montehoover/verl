from urllib.parse import urlparse, urljoin

def parse_and_validate_url(url):
    """
    Parse and validate a URL string.
    
    Args:
        url (str): The URL string to parse and validate
        
    Returns:
        dict: A dictionary containing the URL components (scheme, domain, path, etc.)
        
    Raises:
        ValueError: If the URL is invalid
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")
    
    if not url.strip():
        raise ValueError("URL cannot be empty")
    
    # Parse the URL
    parsed = urlparse(url)
    
    # Validate that we have at least a scheme and netloc (domain)
    if not parsed.scheme:
        raise ValueError("URL must include a scheme (e.g., http://, https://)")
    
    if not parsed.netloc:
        raise ValueError("URL must include a domain")
    
    # Build the result dictionary
    result = {
        'scheme': parsed.scheme,
        'domain': parsed.netloc,
        'path': parsed.path if parsed.path else '/',
        'params': parsed.params,
        'query': parsed.query,
        'fragment': parsed.fragment,
        'username': parsed.username,
        'password': parsed.password,
        'hostname': parsed.hostname,
        'port': parsed.port
    }
    
    return result


def concatenate_url_path(base_url, path):
    """
    Concatenate a path to a base URL.
    
    Args:
        base_url (str): The base URL
        path (str): The path to append
        
    Returns:
        str: The full URL with the path concatenated
    """
    return urljoin(base_url, path)
