from urllib.parse import urlparse

def extract_url_components(url):
    """
    Extract components from a URL.
    
    Args:
        url (str): The URL to parse
        
    Returns:
        dict: A dictionary containing the URL components:
            - scheme: The protocol (e.g., 'http', 'https')
            - netloc: The network location (e.g., 'www.example.com')
            - path: The path (e.g., '/path/to/resource')
            - params: Parameters (semicolon-separated)
            - query: Query string (after the '?')
            - fragment: Fragment identifier (after the '#')
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

INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]

def is_internal_network(url):
    """
    Check if a URL points to an internal or private network.
    
    Args:
        url (str): The URL to check
        
    Returns:
        bool: True if the URL points to an internal network, False otherwise
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
