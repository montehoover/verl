from urllib.parse import urlparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]

def analyze_webhook(webhook_address):
    """
    Processes a webhook URL sent from an external service and returns a validated URL
    that can be safely used for internal API requests.
    
    Args:
        webhook_address: A string representing the URL supplied by an external service.
    
    Returns:
        A validated URL that can be safely used for internal API requests.
    
    Raises:
        ValueError: If the domain doesn't belong to the allowed API endpoint list
                   or if the URL points to an internal/private network.
    """
    # Parse the URL
    parsed_url = urlparse(webhook_address)
    
    # Extract the domain/hostname
    domain = parsed_url.netloc
    
    # Check if the domain is in the approved list
    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the approved API domains list")
    
    # Check if the URL points to internal/private networks
    for internal_ip in INTERNAL_IP_RANGES:
        if domain.startswith(internal_ip):
            raise ValueError(f"URL points to an internal/private network: {domain}")
    
    # Return the validated URL
    return webhook_address
