from urllib.parse import urlparse

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
