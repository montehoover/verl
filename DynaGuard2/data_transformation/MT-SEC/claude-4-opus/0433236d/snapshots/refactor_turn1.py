from urllib.parse import urlparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]

def handle_webhook_request(external_url):
    """
    Process a webhook URL sent from an external service.
    
    Args:
        external_url: A string representing the URL supplied by an external service.
        
    Returns:
        A validated URL that can be safely used for internal API requests.
        
    Raises:
        ValueError: If the domain doesn't belong to the allowed API endpoint list
                   or if the URL points to an internal/private network.
    """
    # Parse the URL
    parsed_url = urlparse(external_url)
    
    # Extract the domain/hostname
    domain = parsed_url.hostname
    
    if not domain:
        raise ValueError("Invalid URL: No domain found")
    
    # Check if domain is in approved list
    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the approved API endpoint list")
    
    # Check if URL points to internal/private network
    for internal_range in INTERNAL_IP_RANGES:
        if domain.startswith(internal_range):
            raise ValueError(f"URL points to internal/private network: {domain}")
    
    # Return the validated URL
    return external_url
