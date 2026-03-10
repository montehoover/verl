from urllib.parse import urlparse

def validate_url_domain(url):
    """
    Validates whether the domain of a URL is in the allowed list.
    
    Args:
        url (str): The URL to validate
        
    Returns:
        bool: True if the domain is allowed, False otherwise
        
    Raises:
        ValueError: If the URL is invalid or malformed
    """
    # Predefined set of allowed domains
    ALLOWED_DOMAINS = {
        'example.com',
        'trusted-site.org',
        'safe-domain.net',
        'internal.company.com'
    }
    
    try:
        # Parse the URL
        parsed = urlparse(url)
        
        # Check if scheme and netloc are present (basic URL validation)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid URL format: {url}")
        
        # Extract the domain (netloc may include port, so we split)
        domain = parsed.netloc.split(':')[0].lower()
        
        # Remove 'www.' prefix if present for comparison
        if domain.startswith('www.'):
            domain = domain[4:]
        
        # Check if domain is in allowed list
        return domain in ALLOWED_DOMAINS
        
    except Exception as e:
        raise ValueError(f"Error parsing URL '{url}': {str(e)}")
