from urllib.parse import urlparse

def extract_domain(url):
    """
    Extract the domain from a URL string.
    
    Args:
        url (str): The URL to extract the domain from
        
    Returns:
        str: The domain part of the URL
    """
    parsed_url = urlparse(url)
    return parsed_url.netloc

def check_safe_domain(url, safe_domains):
    """
    Check if a URL's domain is in the list of safe domains.
    
    Args:
        url (str): The URL to check
        safe_domains (list): List of safe domain strings
        
    Returns:
        str: 'safe' if domain is in safe_domains, 'unsafe' otherwise
    """
    domain = extract_domain(url)
    if domain in safe_domains:
        return 'safe'
    else:
        return 'unsafe'
