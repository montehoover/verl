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
        'fragment': parsed.fragment
    }

def check_trusted_domain(url, trusted_domains):
    """
    Checks if the domain of a URL is in the list of trusted domains.
    
    Args:
        url (str): The URL string to check
        trusted_domains (list): List of trusted domain names
        
    Returns:
        str: Message indicating if the domain is trusted or not
    """
    components = extract_url_components(url)
    domain = components['domain']
    
    if domain in trusted_domains:
        return f"The domain '{domain}' is trusted."
    else:
        return f"The domain '{domain}' is not trusted."
