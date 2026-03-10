from urllib.parse import urlparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]

def is_internal_network(domain):
    """
    Check if a domain points to an internal network location.
    
    Args:
        domain: hostname/domain to check
        
    Returns:
        True if domain is internal, False otherwise
    """
    for internal_range in INTERNAL_IP_RANGES:
        if domain.startswith(internal_range):
            return True
    return False

def is_approved_domain(domain):
    """
    Check if a domain is in the approved API endpoints list.
    
    Args:
        domain: hostname/domain to check
        
    Returns:
        True if domain is approved, False otherwise
    """
    return domain in APPROVED_API_DOMAINS

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
    
    if not domain:
        raise ValueError("Invalid URL: no domain found")
    
    # Check for internal network locations
    if is_internal_network(domain):
        raise ValueError(f"URL points to internal network location: {domain}")
    
    # Check if domain is in approved list
    if not is_approved_domain(domain):
        raise ValueError(f"Domain not in approved API endpoints: {domain}")
    
    # Return the validated URL
    return hook_url
