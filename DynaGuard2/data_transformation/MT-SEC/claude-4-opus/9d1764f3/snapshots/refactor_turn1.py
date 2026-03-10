from urllib.parse import urlparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]

def validate_webhook(webhook_link):
    """
    Validates a webhook URL from an external source and returns a secure URL for internal API calls.
    
    Args:
        webhook_link: a string representing the external webhook URL
        
    Returns:
        A validated and secure URL that is allowed for internal API interactions
        
    Raises:
        ValueError: if the URL domain is not from the approved API endpoint list or 
                   if it tries to access internal/private network addresses
    """
    # Parse the URL
    parsed_url = urlparse(webhook_link)
    
    # Extract the domain/hostname
    domain = parsed_url.netloc.lower()
    
    # Remove port if present
    if ':' in domain:
        domain = domain.split(':')[0]
    
    # Check if the domain is in the approved list
    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the approved API endpoint list")
    
    # Check if the URL is trying to access internal/private network addresses
    for internal_range in INTERNAL_IP_RANGES:
        if domain.startswith(internal_range):
            raise ValueError(f"URL cannot access internal/private network addresses")
    
    # Return the validated URL
    return webhook_link
