from urllib.parse import urlparse

AUTHORIZED_REDIRECT_DOMAINS = ["trusted-redirect.com", "partner.com"]

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

def process_url_redirect(input_url):
    """
    Securely process a URL redirection by validating against approved domains.
    
    Args:
        input_url (str): The URL to validate for redirection
        
    Returns:
        str: The validated URL if domain is approved
        
    Raises:
        ValueError: If the domain is not in the approved list
    """
    parsed = urlparse(input_url)
    domain = parsed.netloc
    
    # Remove port number if present
    if ':' in domain:
        domain = domain.split(':')[0]
    
    if domain not in AUTHORIZED_REDIRECT_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not authorized for redirection")
    
    return input_url
