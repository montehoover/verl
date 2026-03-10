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
        raise ValueError("Invalid URL: no hostname found")
    
    # Check for internal/private network addresses
    for internal_prefix in INTERNAL_IP_RANGES:
        if hostname.startswith(internal_prefix):
            raise ValueError(f"URL attempts to access internal/private network: {hostname}")
    
    # Check if domain is in approved list
    if hostname not in APPROVED_API_DOMAINS:
        raise ValueError(f"URL domain not in approved list: {hostname}")
    
    # Ensure the URL uses HTTPS for security
    if parsed.scheme != 'https':
        raise ValueError("URL must use HTTPS protocol")
    
    # Return the validated URL
    return callback_url
