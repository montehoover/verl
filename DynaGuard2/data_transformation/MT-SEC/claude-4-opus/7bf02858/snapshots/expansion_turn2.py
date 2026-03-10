from urllib.parse import urlparse

INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]

def extract_url_components(url):
    """
    Extract components from a URL.
    
    Args:
        url (str): The URL to parse
        
    Returns:
        dict: A dictionary containing the URL components:
            - scheme: The protocol (e.g., 'http', 'https')
            - netloc: The network location (domain and port)
            - path: The path component
            - params: Parameters (rarely used)
            - query: Query string
            - fragment: Fragment identifier (after #)
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

def is_internal_url(url):
    """
    Check if a URL is targeting an internal network.
    
    Args:
        url (str): The URL to check
        
    Returns:
        bool: True if the URL is internal, False otherwise
    """
    parsed = urlparse(url)
    netloc = parsed.netloc.lower()
    
    # Remove port if present
    if ':' in netloc:
        netloc = netloc.split(':')[0]
    
    # Check against internal IP patterns
    for pattern in INTERNAL_IP_RANGES:
        if netloc.startswith(pattern):
            return True
    
    return False
