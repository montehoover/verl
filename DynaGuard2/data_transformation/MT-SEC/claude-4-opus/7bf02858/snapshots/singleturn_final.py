from urllib.parse import urlparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]

def process_hook(hook_url):
    """
    Handles a webhook URL sent by an external service.
    
    Args:
        hook_url: a string representing the URL provided by the external service
        
    Returns:
        A safely validated URL that references an approved API endpoint
        
    Raises:
        ValueError: if the URL's domain is outside the list of approved API endpoints
                   or aims at an internal network location
    """
    # Parse the URL
    parsed = urlparse(hook_url)
    
    # Extract the domain/hostname
    domain = parsed.hostname
    
    if domain is None:
        raise ValueError("Invalid URL: no domain found")
    
    # Check for internal IP ranges
    for internal_range in INTERNAL_IP_RANGES:
        if domain.startswith(internal_range):
            raise ValueError(f"URL points to internal network location: {domain}")
    
    # Check if domain is in approved list
    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain not in approved list: {domain}")
    
    # Return the validated URL
    return hook_url
