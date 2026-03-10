from urllib.parse import urlparse

def validate_cdn_url(url):
    """
    Validates if a URL's domain is in the list of allowed CDN domains.
    
    Args:
        url (str): The URL to validate
        
    Raises:
        ValueError: If the URL's domain is not in the allowed list
        
    Returns:
        bool: True if the URL is valid
    """
    # List of allowed CDN domains
    allowed_cdn_domains = [
        'cdn.example.com',
        'static.example.com',
        'assets.example.com',
        'cloudfront.net',
        'cdn.jsdelivr.net',
        'unpkg.com',
        'cdnjs.cloudflare.com'
    ]
    
    try:
        # Parse the URL
        parsed_url = urlparse(url)
        
        # Extract the domain (netloc)
        domain = parsed_url.netloc.lower()
        
        # Check if domain is in allowed list
        if domain not in allowed_cdn_domains:
            raise ValueError(f"Domain '{domain}' is not in the allowed CDN domains list")
            
        return True
        
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        else:
            raise ValueError(f"Invalid URL format: {str(e)}")
