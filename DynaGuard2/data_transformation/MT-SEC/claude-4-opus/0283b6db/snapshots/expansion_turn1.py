from urllib.parse import urlparse

def validate_return_url(url):
    """
    Validates if a URL's domain is in the list of allowed domains.
    
    Args:
        url (str): The URL to validate
        
    Returns:
        bool: True if the domain is allowed
        
    Raises:
        ValueError: If the domain is not in the allowed list
    """
    # List of allowed domains
    allowed_domains = [
        'example.com',
        'www.example.com',
        'app.example.com',
        'trusted-site.org',
        'www.trusted-site.org'
    ]
    
    # Parse the URL
    parsed_url = urlparse(url)
    
    # Extract the domain (netloc)
    domain = parsed_url.netloc
    
    # Check if domain is in allowed list
    if domain in allowed_domains:
        return True
    else:
        raise ValueError(f"Domain '{domain}' is not in the allowed domains list")
