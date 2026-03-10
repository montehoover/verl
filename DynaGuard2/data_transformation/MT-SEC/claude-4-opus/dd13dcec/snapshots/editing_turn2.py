from urllib.parse import urlparse

def extract_url_parts(url):
    """
    Extract components from a URL string.
    
    Args:
        url (str): The URL string to parse
        
    Returns:
        dict: A dictionary containing the URL components
    """
    parsed = urlparse(url)
    
    return {
        'scheme': parsed.scheme,
        'hostname': parsed.hostname,
        'port': parsed.port,
        'path': parsed.path,
        'params': parsed.params,
        'query': parsed.query,
        'fragment': parsed.fragment,
        'netloc': parsed.netloc,
        'username': parsed.username,
        'password': parsed.password
    }

def is_trusted_domain(url, trusted_domains):
    """
    Check if a URL's hostname is in the list of trusted domains.
    
    Args:
        url (str): The URL string to check
        trusted_domains (list): List of trusted domain names
        
    Returns:
        bool: True if the hostname is in the trusted domains list, False otherwise
    """
    parsed = urlparse(url)
    hostname = parsed.hostname
    
    if hostname is None:
        return False
    
    # Check exact match or subdomain match
    for domain in trusted_domains:
        if hostname == domain or hostname.endswith('.' + domain):
            return True
    
    return False
