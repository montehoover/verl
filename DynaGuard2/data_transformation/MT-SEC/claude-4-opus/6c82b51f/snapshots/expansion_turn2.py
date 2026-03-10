from urllib.parse import urlparse
import re

INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]

def extract_url_parts(url):
    """
    Parse a URL and return its components.
    
    Args:
        url (str): The URL to parse
        
    Returns:
        dict: A dictionary containing the URL components:
            - scheme: The protocol (e.g., 'http', 'https')
            - netloc: The network location (e.g., 'www.example.com')
            - path: The path (e.g., '/path/to/page')
            - params: Parameters (rarely used)
            - query: Query string (e.g., 'id=123&name=test')
            - fragment: Fragment identifier (e.g., 'section1')
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

def is_internal_network_url(url):
    """
    Check if a URL points to an internal network.
    
    Args:
        url (str): The URL to check
        
    Returns:
        bool: True if the URL points to an internal network, False otherwise
    """
    parsed = urlparse(url)
    host = parsed.hostname
    
    if not host:
        return False
    
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, host):
            return True
    
    return False
