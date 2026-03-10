from urllib.parse import urlparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]

def validate_callback(callback_url):
    """
    Validates a webhook URL from an external source.
    
    Args:
        callback_url: a string representing the external webhook URL
        
    Returns:
        A validated and secure URL that is allowed for internal API interactions
        
    Raises:
        ValueError: if the URL domain is not from the approved API endpoint list 
                   or if it tries to access internal/private network addresses
    """
    # Parse the URL
    parsed = urlparse(callback_url)
    
    # Extract the hostname
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("Invalid URL: No hostname found")
    
    # Check if the hostname is in internal IP ranges
    for internal_range in INTERNAL_IP_RANGES:
        if hostname.startswith(internal_range):
            raise ValueError(f"URL not allowed: Cannot access internal/private network address {hostname}")
    
    # Check if the hostname is in the approved domains list
    if hostname not in APPROVED_API_DOMAINS:
        raise ValueError(f"URL not allowed: Domain {hostname} is not in the approved API endpoint list")
    
    # Return the validated URL
    return callback_url
