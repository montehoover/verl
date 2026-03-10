from urllib.parse import urlparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]

def handle_webhook_url(url):
    """
    Parse a webhook URL and return its components.
    
    Args:
        url (str): The webhook URL to parse
        
    Returns:
        dict: A dictionary containing the domain and path components
    """
    parsed = urlparse(url)
    
    return {
        'domain': parsed.netloc,
        'path': parsed.path,
        'scheme': parsed.scheme,
        'query': parsed.query,
        'fragment': parsed.fragment
    }

def verify_webhook_domain(url):
    """
    Check if a URL's domain is in the list of approved domains.
    
    Args:
        url (str): The webhook URL to verify
        
    Returns:
        bool: True if the domain is approved, False otherwise
    """
    parsed = urlparse(url)
    return parsed.netloc in APPROVED_API_DOMAINS

def validate_webhook(url):
    """
    Validate a webhook URL is from an approved domain and not accessing internal networks.
    
    Args:
        url (str): The webhook URL to validate
        
    Returns:
        str: The secure URL if valid
        
    Raises:
        ValueError: If the URL is not from an approved domain or attempts to access internal addresses
    """
    parsed = urlparse(url)
    domain = parsed.netloc
    
    # Check if domain is approved
    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the approved domains list")
    
    # Check for internal IP addresses
    for internal_range in INTERNAL_IP_RANGES:
        if domain.startswith(internal_range):
            raise ValueError(f"URL attempts to access internal network: {domain}")
    
    return url
