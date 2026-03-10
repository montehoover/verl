from urllib.parse import urlparse

def extract_url_components(url):
    """
    Extracts components from a URL string.
    
    Args:
        url (str): The URL string to parse
        
    Returns:
        dict: A dictionary containing the URL components
    """
    parsed = urlparse(url)
    
    return {
        'scheme': parsed.scheme,
        'domain': parsed.netloc,
        'path': parsed.path,
        'params': parsed.params,
        'query': parsed.query,
        'fragment': parsed.fragment,
        'port': parsed.port,
        'username': parsed.username,
        'password': parsed.password
    }
