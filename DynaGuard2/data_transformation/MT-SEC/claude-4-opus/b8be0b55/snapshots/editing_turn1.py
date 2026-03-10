from urllib.parse import urlparse

def extract_url_parts(url):
    """
    Extract components from a URL string.
    
    Args:
        url (str): The URL to parse
        
    Returns:
        dict: Dictionary containing URL components
    """
    parsed = urlparse(url)
    
    return {
        'scheme': parsed.scheme,
        'netloc': parsed.netloc,
        'path': parsed.path,
        'params': parsed.params,
        'query': parsed.query,
        'fragment': parsed.fragment
    }
