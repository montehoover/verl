from urllib.parse import urlparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]

def extract_url_components(url):
    """
    Extract components from a URL string.
    
    Args:
        url (str): The URL to parse
        
    Returns:
        dict: Dictionary containing URL components with keys:
            - domain: The domain/hostname
            - path: The path component
            - scheme: The protocol (http, https, etc.)
            - port: The port number (if specified)
            - query: The query string (if present)
            - fragment: The fragment/anchor (if present)
    """
    parsed = urlparse(url)
    
    return {
        'domain': parsed.netloc,
        'path': parsed.path,
        'scheme': parsed.scheme,
        'port': parsed.port,
        'query': parsed.query,
        'fragment': parsed.fragment
    }


def check_domain_approval(url, allowed_domains):
    """
    Check if a URL's domain is in the list of allowed domains.
    
    Args:
        url (str): The URL to check
        allowed_domains (list): List of approved domain names
        
    Returns:
        dict: Dictionary containing:
            - domain: The extracted domain
            - status: 'approved' or 'not approved'
            - message: Status message
    """
    components = extract_url_components(url)
    domain = components['domain']
    
    # Remove port if present for comparison
    domain_without_port = domain.split(':')[0]
    
    if domain_without_port in allowed_domains:
        status = 'approved'
        message = f"Domain '{domain}' is approved"
    else:
        status = 'not approved'
        message = f"Domain '{domain}' is not approved"
    
    return {
        'domain': domain,
        'status': status,
        'message': message
    }


def handle_webhook_request(external_url):
    """
    Process a webhook URL from an external service and validate it for internal API requests.
    
    Args:
        external_url (str): The webhook URL to process
        
    Returns:
        str: The validated URL
        
    Raises:
        ValueError: If the domain isn't approved or if URL points to internal/private network
    """
    parsed = urlparse(external_url)
    domain = parsed.netloc
    
    # Remove port if present for comparison
    domain_without_port = domain.split(':')[0]
    
    # Check if domain points to internal/private network
    for internal_range in INTERNAL_IP_RANGES:
        if domain_without_port.startswith(internal_range):
            raise ValueError(f"URL points to internal/private network: {domain}")
    
    # Check if domain is in approved list
    if domain_without_port not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain not approved: {domain}")
    
    # Return the validated URL
    return external_url
