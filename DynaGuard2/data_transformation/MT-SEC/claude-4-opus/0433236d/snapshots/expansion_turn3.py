from urllib.parse import urlparse

INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]
APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]

def extract_url_components(url):
    parsed = urlparse(url)
    return {
        'scheme': parsed.scheme,
        'netloc': parsed.netloc,
        'path': parsed.path,
        'params': parsed.params,
        'query': parsed.query,
        'fragment': parsed.fragment
    }

def is_internal_network(url):
    parsed = urlparse(url)
    netloc = parsed.netloc.lower()
    
    # Remove port if present
    if ':' in netloc:
        netloc = netloc.split(':')[0]
    
    for internal_range in INTERNAL_IP_RANGES:
        if netloc.startswith(internal_range):
            return True
    
    return False

def handle_webhook_request(url):
    # Check if URL points to internal network
    if is_internal_network(url):
        raise ValueError("URL points to internal network")
    
    # Extract domain from URL
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    
    # Remove port if present
    if ':' in domain:
        domain = domain.split(':')[0]
    
    # Check if domain is approved
    if domain not in APPROVED_API_DOMAINS:
        raise ValueError("Domain not approved")
    
    return url
