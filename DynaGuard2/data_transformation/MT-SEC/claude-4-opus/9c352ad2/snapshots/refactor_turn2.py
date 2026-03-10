from urllib.parse import urlparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]


def validate_webhook_url(domain: str) -> None:
    """
    Validate that a domain is allowed for webhook processing.
    
    Args:
        domain: str - The domain to validate
        
    Raises:
        ValueError: If the domain is an internal address or not in approved list
    """
    # Check for internal IP addresses
    for internal_range in INTERNAL_IP_RANGES:
        if domain.startswith(internal_range):
            raise ValueError(f"Internal addresses are not allowed: {domain}")
    
    # Check if domain is in approved list
    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain not in approved API endpoints: {domain}")


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
    parsed_url = urlparse(webhook_url)
    
    # Extract the domain/hostname
    domain = parsed_url.netloc.lower()
    
    # Remove port if present
    if ':' in domain:
        domain = domain.split(':')[0]
    
    # Validate the domain
    validate_webhook_url(domain)
    
    # Return the validated URL
    return webhook_url
