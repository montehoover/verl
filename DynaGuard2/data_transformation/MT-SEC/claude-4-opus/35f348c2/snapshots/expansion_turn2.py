def is_http_protocol(url):
    """Check if a string starts with 'http' or 'https'.
    
    Args:
        url (str): The string to check
        
    Returns:
        bool: True if the string starts with 'http' or 'https', False otherwise
    """
    if not isinstance(url, str):
        return False
    
    return url.startswith('http://') or url.startswith('https://')


def extract_url_components(url):
    """Extract protocol, domain, and path components from a URL.
    
    Args:
        url (str): The URL string to parse
        
    Returns:
        dict: A dictionary with keys 'protocol', 'domain', and 'path'
    """
    if not isinstance(url, str):
        return {'protocol': '', 'domain': '', 'path': ''}
    
    # Find protocol
    protocol_end = url.find('://')
    if protocol_end == -1:
        return {'protocol': '', 'domain': '', 'path': ''}
    
    protocol = url[:protocol_end]
    
    # Find domain
    domain_start = protocol_end + 3
    domain_end = url.find('/', domain_start)
    
    if domain_end == -1:
        # No path, entire rest is domain
        domain = url[domain_start:]
        path = ''
    else:
        domain = url[domain_start:domain_end]
        path = url[domain_end:]
    
    return {
        'protocol': protocol,
        'domain': domain,
        'path': path
    }
