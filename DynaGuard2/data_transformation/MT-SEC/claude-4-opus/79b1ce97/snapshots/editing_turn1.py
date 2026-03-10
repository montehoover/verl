from urllib.parse import urlparse

# Predefined set of approved domains
APPROVED_DOMAINS = {
    'example.com',
    'trusted-site.org',
    'safe-domain.net',
    'approved-host.io'
}

def validate_url_domain(url):
    """
    Validates if a URL's domain is in the approved domains set.
    
    Args:
        url (str): The URL to validate
        
    Raises:
        ValueError: If the domain is not in the approved domains set
        
    Returns:
        bool: True if the domain is approved
    """
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        
        # Remove www. prefix if present
        if domain.startswith('www.'):
            domain = domain[4:]
            
        if domain not in APPROVED_DOMAINS:
            raise ValueError(f"Domain '{domain}' is not in the approved domains list")
            
        return True
        
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        else:
            raise ValueError(f"Invalid URL format: {url}")
