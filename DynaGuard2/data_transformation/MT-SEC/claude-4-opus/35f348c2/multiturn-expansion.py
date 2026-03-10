import re


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


def validate_path(path):
    """Validate if a string is a well-formed http or https URL.
    
    Args:
        path (str): The string to validate
        
    Returns:
        bool: True if the input is a valid http/https URL, False otherwise
    """
    try:
        if not isinstance(path, str):
            return False
        
        # Regular expression for validating http/https URLs
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)*(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|'  # ...or ipv4
            r'\[?[A-F0-9]*:[A-F0-9:]+\]?)'  # ...or ipv6
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        return bool(url_pattern.match(path))
    except:
        return False
