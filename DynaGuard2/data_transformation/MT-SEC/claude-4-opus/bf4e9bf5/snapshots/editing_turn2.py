from urllib.parse import urlparse

def extract_url_components(url):
    """
    Extracts components from a URL string.
    
    Args:
        url (str): The URL string to parse
        
    Returns:
        dict: A dictionary containing the URL components
    """
    parsed = urlparse(url)
    
    return {
        'scheme': parsed.scheme,
        'domain': parsed.netloc,
        'path': parsed.path,
        'params': parsed.params,
        'query': parsed.query,
        'fragment': parsed.fragment,
        'port': parsed.port,
        'username': parsed.username,
        'password': parsed.password
    }

def check_url_validity(url, allowed_domains):
    """
    Checks if a URL is valid based on scheme and domain criteria.
    
    Args:
        url (str): The URL string to validate
        allowed_domains (list): List of allowed domain names
        
    Returns:
        bool: True if URL has http/https scheme and domain is in allowed list, False otherwise
    """
    parsed = urlparse(url)
    
    # Check if scheme is http or https
    if parsed.scheme not in ['http', 'https']:
        return False
    
    # Check if domain is in allowed list
    if parsed.netloc not in allowed_domains:
        return False
    
    return True
