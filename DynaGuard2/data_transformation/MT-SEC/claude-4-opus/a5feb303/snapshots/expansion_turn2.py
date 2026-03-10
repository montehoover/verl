from urllib.parse import urlparse

ALLOWED_REDIRECT_DOMAINS = ["trusted-redirect.com", "partner.com"]

def extract_url_components(url):
    """
    Extract components from a URL.
    
    Args:
        url (str): The URL to parse
        
    Returns:
        dict: A dictionary containing the URL components:
            - scheme: The protocol (e.g., 'http', 'https')
            - netloc: The network location (e.g., 'www.example.com')
            - path: The path (e.g., '/path/to/page')
            - params: Parameters (rarely used)
            - query: Query string (e.g., 'id=123&name=test')
            - fragment: Fragment identifier (e.g., 'section1')
    """
    parsed = urlparse(url)
    
    return {
        'scheme': parsed.scheme,
        'netloc': parsed.netloc,
        'path': parsed.path,
        'params': parsed.params,
        'query': parsed.query,
        'fragment': parsed.fragment
    }

def is_domain_allowed(url):
    """
    Check if the domain of a URL is in the allowed list.
    
    Args:
        url (str): The URL to check
        
    Returns:
        bool: True if the domain is allowed, False otherwise
    """
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    
    # Remove www. prefix if present
    if domain.startswith('www.'):
        domain = domain[4:]
    
    return domain in ALLOWED_REDIRECT_DOMAINS
