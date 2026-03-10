from urllib.parse import urlparse

def validate_url_domain(url):
    """
    Validate if a URL's domain is in the allowed list.
    
    Args:
        url (str): The URL to validate
        
    Returns:
        bool: True if domain is allowed, False otherwise
        
    Raises:
        ValueError: If the URL is invalid
    """
    # Predefined list of allowed domains
    ALLOWED_DOMAINS = [
        'example.com',
        'trusted-site.org',
        'safe-domain.net',
        'localhost'
    ]
    
    try:
        # Parse the URL
        parsed = urlparse(url)
        
        # Check if the URL has a valid scheme and netloc
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid URL: {url}")
        
        # Extract the domain (netloc may include port)
        domain = parsed.netloc.split(':')[0].lower()
        
        # Check if domain is in allowed list
        return domain in ALLOWED_DOMAINS
        
    except Exception as e:
        raise ValueError(f"Invalid URL: {url}") from e
