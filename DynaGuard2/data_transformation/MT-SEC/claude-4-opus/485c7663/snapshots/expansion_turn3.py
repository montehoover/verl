from urllib.parse import urlparse
import re

INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]

ALLOWED_API_DOMAINS = ["api.trustedsource.com", "gateway.partnerapi.com"]

def parse_url(url):
    """
    Parse a URL and return its components.
    
    Args:
        url (str): The URL to parse
        
    Returns:
        dict: A dictionary containing the URL components:
            - scheme: The protocol (e.g., 'http', 'https')
            - netloc: The network location (domain and port)
            - path: The path component
            - params: Parameters (semicolon-separated)
            - query: Query string
            - fragment: Fragment identifier
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

def check_internal_access(url):
    """
    Check if a URL points to an internal address.
    
    Args:
        url (str): The URL to check
        
    Returns:
        bool: True if the URL points to an internal address, False otherwise
    """
    parsed = urlparse(url)
    hostname = parsed.hostname
    
    if not hostname:
        return False
    
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            return True
    
    return False

def enforce_api_security(api_endpoint_url):
    """
    Verify that an API endpoint URL is from an approved list.
    
    Args:
        api_endpoint_url (str): The API endpoint URL to verify
        
    Returns:
        str: The verified URL if secure
        
    Raises:
        ValueError: If the URL is not from an approved domain
    """
    parsed = urlparse(api_endpoint_url)
    hostname = parsed.hostname
    
    if not hostname:
        raise ValueError("Invalid URL: no hostname found")
    
    if hostname not in ALLOWED_API_DOMAINS:
        raise ValueError(f"Unauthorized API domain: {hostname}")
    
    return api_endpoint_url
