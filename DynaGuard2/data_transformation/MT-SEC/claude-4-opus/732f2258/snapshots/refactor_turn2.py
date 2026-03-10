from urllib.parse import urlparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]

def is_approved_domain(domain):
    """Check if the domain is in the approved API domains list."""
    return domain in APPROVED_API_DOMAINS

def is_internal_network(domain):
    """Check if the domain points to an internal or private network."""
    for internal_range in INTERNAL_IP_RANGES:
        if domain.startswith(internal_range):
            return True
    return False

def analyze_webhook(webhook_address):
    # Parse the URL
    parsed_url = urlparse(webhook_address)
    
    # Extract the domain/hostname
    domain = parsed_url.hostname
    
    if not domain:
        raise ValueError("Invalid URL: No domain found")
    
    # Check if domain is in approved list
    if not is_approved_domain(domain):
        raise ValueError(f"Domain '{domain}' is not in the approved API endpoint list")
    
    # Check if the URL points to internal/private network
    if is_internal_network(domain):
        raise ValueError(f"URL points to internal/private network: {domain}")
    
    # Return the validated URL
    return webhook_address
