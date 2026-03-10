from urllib.parse import urlparse

def extract_url_components(url):
    """
    Extract components from a URL string.
    
    Args:
        url (str): The URL to parse
        
    Returns:
        dict: Dictionary containing URL components with keys:
            - domain: The domain/hostname
            - path: The path portion of the URL
            - scheme: The protocol (http, https, etc.)
            - port: The port number (if specified)
            - query: The query string (if present)
            - fragment: The fragment/anchor (if present)
    """
    parsed = urlparse(url)
    
    return {
        'domain': parsed.netloc,
        'path': parsed.path,
        'scheme': parsed.scheme,
        'port': parsed.port,
        'query': parsed.query,
        'fragment': parsed.fragment
    }
