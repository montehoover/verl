from urllib.parse import urlparse

def extract_url_parts(url):
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
        'netloc': parsed.netloc,
        'path': parsed.path,
        'params': parsed.params,
        'query': parsed.query,
        'fragment': parsed.fragment
    }

def is_valid_url(url):
    """
    Check if a URL is valid based on scheme and domain criteria.
    
    Args:
        url (str): The URL string to validate
        
    Returns:
        bool: True if URL has http/https scheme and non-localhost domain
    """
    parsed = urlparse(url)
    
    # Check if scheme is http or https
    if parsed.scheme not in ['http', 'https']:
        return False
    
    # Check if domain is localhost or loopback
    domain = parsed.netloc.lower()
    if not domain:
        return False
    
    # Remove port if present
    domain = domain.split(':')[0]
    
    # Check for localhost variations and loopback addresses
    invalid_domains = ['localhost', '127.0.0.1', '[::1]', '::1']
    if domain in invalid_domains:
        return False
    
    # Check for loopback IP range (127.0.0.0/8)
    if domain.startswith('127.'):
        parts = domain.split('.')
        if len(parts) == 4 and all(part.isdigit() for part in parts):
            return False
    
    return True
