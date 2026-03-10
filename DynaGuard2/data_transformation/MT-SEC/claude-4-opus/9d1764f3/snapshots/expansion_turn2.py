from urllib.parse import urlparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]

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
