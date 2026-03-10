from urllib.parse import urlparse

INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]
APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]

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

def process_hook(url):
    """
    Validate and process a webhook URL.
    
    Args:
        url (str): The URL to validate
        
    Returns:
        str: The validated URL
        
    Raises:
        ValueError: If the URL is not from an approved domain or targets internal network
    """
    # Check if URL targets internal network
    if is_internal_url(url):
        raise ValueError("URL targets internal network")
    
    # Parse the URL to extract domain
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    
    # Remove port if present
    if ':' in domain:
        domain = domain.split(':')[0]
    
    # Check if domain is in approved list
    if domain not in APPROVED_API_DOMAINS:
        raise ValueError("URL is not from an approved domain")
    
    return url
