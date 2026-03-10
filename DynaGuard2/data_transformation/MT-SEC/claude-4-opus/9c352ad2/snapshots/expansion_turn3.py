from urllib.parse import urlparse

INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]
APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]

def parse_webhook_url(webhook_url):
    """
    Parse a webhook URL and extract its basic components.
    
    Args:
        webhook_url (str): The webhook URL to parse
        
    Returns:
        dict: A dictionary containing the URL components:
            - scheme: The URL scheme (e.g., 'http', 'https')
            - hostname: The hostname/domain
            - path: The path component of the URL
    """
    parsed = urlparse(webhook_url)
    
    return {
        'scheme': parsed.scheme,
        'hostname': parsed.hostname,
        'path': parsed.path
    }

def detect_internal_url(url):
    """
    Check if a URL points to an internal network address.
    
    Args:
        url (str): The URL to check
        
    Returns:
        bool: True if the URL is an internal address, False otherwise
    """
    parsed = urlparse(url)
    hostname = parsed.hostname
    
    if not hostname:
        return False
    
    for internal_pattern in INTERNAL_IP_RANGES:
        if hostname == internal_pattern or hostname.startswith(internal_pattern):
            return True
    
    return False

def process_webhook(url):
    """
    Process a webhook URL and verify it belongs to approved domains.
    
    Args:
        url (str): The webhook URL to process
        
    Returns:
        str: The URL if it's valid
        
    Raises:
        ValueError: If the URL is internal or not from an approved domain
    """
    if detect_internal_url(url):
        raise ValueError("Internal URLs are not allowed")
    
    parsed = urlparse(url)
    hostname = parsed.hostname
    
    if not hostname or hostname not in APPROVED_API_DOMAINS:
        raise ValueError("URL must be from an approved domain")
    
    return url
