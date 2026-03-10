from urllib.parse import urlparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]

def extract_domain(url):
    """
    Extracts the domain from a URL, removing any port information.
    
    Args:
        url: a string representing a URL
        
    Returns:
        The domain/hostname in lowercase without port
    """
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()
    
    # Remove port if present
    if ':' in domain:
        domain = domain.split(':')[0]
    
    return domain

def is_approved_domain(domain):
    """
    Checks if a domain is in the approved API endpoint list.
    
    Args:
        domain: a string representing the domain to check
        
    Returns:
        True if the domain is approved, False otherwise
    """
    return domain in APPROVED_API_DOMAINS

def is_internal_address(domain):
    """
    Checks if a domain represents an internal/private network address.
    
    Args:
        domain: a string representing the domain to check
        
    Returns:
        True if the domain is an internal address, False otherwise
    """
    for internal_range in INTERNAL_IP_RANGES:
        if domain.startswith(internal_range):
            return True
    return False

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
    # Extract the domain from the URL
    domain = extract_domain(webhook_link)
    
    # Check if the domain is in the approved list
    if not is_approved_domain(domain):
        raise ValueError(f"Domain '{domain}' is not in the approved API endpoint list")
    
    # Check if the URL is trying to access internal/private network addresses
    if is_internal_address(domain):
        raise ValueError(f"URL cannot access internal/private network addresses")
    
    # Return the validated URL
    return webhook_link
