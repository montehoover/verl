from urllib.parse import urlparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]

def process_webhook(webhook_url: str) -> str:
    """
    Process a webhook URL provided by an external service.
    
    Args:
        webhook_url: str - The URL provided by an external service
        
    Returns:
        A validated URL that points to an approved API endpoint
        
    Raises:
        ValueError: If the URL's domain does not belong to pre-approved API endpoints
                   or an internal address is attempted
    """
    # Parse the URL
    parsed = urlparse(webhook_url)
    
    # Extract the domain/hostname
    domain = parsed.hostname
    
    # Check if domain is None (invalid URL)
    if domain is None:
        raise ValueError("Invalid URL format")
    
    # Check for internal IP addresses
    for internal_range in INTERNAL_IP_RANGES:
        if domain.startswith(internal_range):
            raise ValueError(f"Internal addresses are not allowed: {domain}")
    
    # Check if domain is in approved list
    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the approved API endpoints list")
    
    # Return the validated URL
    return webhook_url
