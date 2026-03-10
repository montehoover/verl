from urllib.parse import urlparse

def validate_domain(url):
    """
    Validates if the domain of the given URL is in the list of approved domains.
    
    Args:
        url (str): The URL to validate
        
    Returns:
        bool: True if the domain is approved
        
    Raises:
        ValueError: If the domain is not in the approved list
    """
    # List of approved domains
    APPROVED_DOMAINS = [
        'example.com',
        'trusted-site.org',
        'safe-domain.net',
        'approved.io'
    ]
    
    # Parse the URL to extract the domain
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()
    
    # Remove www. prefix if present
    if domain.startswith('www.'):
        domain = domain[4:]
    
    # Check if domain is in approved list
    if domain in APPROVED_DOMAINS:
        return True
    else:
        raise ValueError(f"Domain '{domain}' is not in the list of approved domains")
