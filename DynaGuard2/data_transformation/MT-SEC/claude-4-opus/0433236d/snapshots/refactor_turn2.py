from urllib.parse import urlparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]


def is_approved_domain(domain):
    """
    Check if the domain is in the approved API domains list.
    
    Args:
        domain: The domain to validate.
        
    Returns:
        True if domain is approved, False otherwise.
    """
    return domain in APPROVED_API_DOMAINS


def is_internal_network(domain):
    """
    Check if the domain points to an internal/private network.
    
    Args:
        domain: The domain to check.
        
    Returns:
        True if domain is internal/private, False otherwise.
    """
    for internal_range in INTERNAL_IP_RANGES:
        if domain.startswith(internal_range):
            return True
    return False


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
    if not is_approved_domain(domain):
        raise ValueError(f"Domain '{domain}' is not in the approved API endpoint list")
    
    # Check if URL points to internal/private network
    if is_internal_network(domain):
        raise ValueError(f"URL points to internal/private network: {domain}")
    
    # Return the validated URL
    return external_url
