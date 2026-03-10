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
            - path: The path portion of the URL
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

def check_approved_domain(url, approved_domains):
    """
    Check if a URL's domain is in the list of approved domains.
    
    Args:
        url (str): The URL to check
        approved_domains (list): List of approved domain names
        
    Returns:
        tuple: (bool, str) - (is_approved, message)
            - is_approved: True if domain is in approved list, False otherwise
            - message: 'approved' or 'not approved'
    """
    components = extract_url_components(url)
    domain = components['domain']
    
    # Remove port from domain if present
    if ':' in domain:
        domain = domain.split(':')[0]
    
    is_approved = domain in approved_domains
    message = 'approved' if is_approved else 'not approved'
    
    return is_approved, message

def process_hook(hook_url):
    """
    Process a webhook URL to ensure it's from an approved API domain
    and not targeting internal networks.
    
    Args:
        hook_url (str): The webhook URL to validate
        
    Returns:
        str: The validated URL if it meets all criteria
        
    Raises:
        ValueError: If the URL doesn't meet security criteria
    """
    components = extract_url_components(hook_url)
    domain = components['domain']
    
    # Remove port from domain if present
    if ':' in domain:
        domain_without_port = domain.split(':')[0]
    else:
        domain_without_port = domain
    
    # Check if domain is in approved list
    if domain_without_port not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{domain_without_port}' is not in the approved API domains list")
    
    # Check if domain targets internal networks
    for internal_range in INTERNAL_IP_RANGES:
        if domain_without_port.startswith(internal_range):
            raise ValueError(f"Domain '{domain_without_port}' targets internal network")
    
    return hook_url
