from urllib.parse import urlparse

TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]

def parse_image_url(url):
    """
    Parse an image URL and return its components.
    
    Args:
        url (str): The image URL to parse
        
    Returns:
        tuple: A tuple containing (scheme, netloc, path, params, query, fragment)
    """
    parsed = urlparse(url)
    return (parsed.scheme, parsed.netloc, parsed.path, 
            parsed.params, parsed.query, parsed.fragment)

def is_trusted_domain(domain):
    """
    Check if a domain belongs to the list of trusted image hosting services.
    
    Args:
        domain (str): The domain to check
        
    Returns:
        bool: True if the domain is trusted, False otherwise
    """
    return domain in TRUSTED_IMAGE_HOSTS
