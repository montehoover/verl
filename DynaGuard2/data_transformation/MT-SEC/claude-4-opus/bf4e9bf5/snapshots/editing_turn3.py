from urllib.parse import urlparse

TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
ALLOWED_SCHEMES = ["http", "https"]

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

def resolve_image_url(image_url):
    """
    Ensures an image URL belongs to a trusted image hosting service.
    
    Args:
        image_url (str): The image URL to validate
        
    Returns:
        str: The resolved URL if from a trusted domain
        
    Raises:
        ValueError: If the URL's domain is not in the trusted list
    """
    parsed = urlparse(image_url)
    
    # Check if scheme is allowed
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"URL scheme must be one of {ALLOWED_SCHEMES}")
    
    # Check if domain is trusted
    if parsed.netloc not in TRUSTED_IMAGE_HOSTS:
        raise ValueError(f"Image host '{parsed.netloc}' is not in the list of trusted hosts")
    
    return image_url
