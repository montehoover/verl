from urllib.parse import urlparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]

def validate_webhook(webhook_link):
    """
    Validates a webhook URL from an external source.
    
    Args:
        webhook_link: a string representing the external webhook URL
        
    Returns:
        A validated and secure URL that is allowed for internal API interactions
        
    Raises:
        ValueError: if the URL domain is not from the approved API endpoint list 
                   or if it tries to access internal/private network addresses
    """
    # Parse the URL
    parsed_url = urlparse(webhook_link)
    
    # Extract the hostname
    hostname = parsed_url.hostname
    
    if not hostname:
        raise ValueError("Invalid URL: No hostname found")
    
    # Check for internal/private network addresses
    for internal_range in INTERNAL_IP_RANGES:
        if hostname.startswith(internal_range):
            raise ValueError(f"Access to internal/private network address '{hostname}' is not allowed")
    
    # Check if the domain is in the approved list
    if hostname not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{hostname}' is not in the approved API endpoint list")
    
    # Return the validated URL
    return webhook_link
