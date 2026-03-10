from urllib.parse import urlparse

def extract_url_components(url):
    """
    Extract components from a URL string.
    
    Args:
        url (str): The URL string to parse
        
    Returns:
        dict: Dictionary containing URL components
    """
    parsed = urlparse(url)
    
    return {
        'scheme': parsed.scheme,
        'domain': parsed.netloc,
        'path': parsed.path,
        'params': parsed.params,
        'query': parsed.query,
        'fragment': parsed.fragment
    }

def check_trusted_domain(url, trusted_domains=None):
    """
    Check if a URL's domain is in the list of trusted domains.
    
    Args:
        url (str): The URL string to check
        trusted_domains (list): List of trusted domain names
        
    Returns:
        str: 'trusted' if domain is in trusted list, 'untrusted' otherwise
    """
    if trusted_domains is None:
        trusted_domains = []
    
    components = extract_url_components(url)
    domain = components['domain']
    
    # Remove port number if present
    if ':' in domain:
        domain = domain.split(':')[0]
    
    return 'trusted' if domain in trusted_domains else 'untrusted'
