from urllib.parse import urlparse

TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
ALLOWED_SCHEMES = ["http", "https"]

def parse_image_url(url):
    """
    Parse an image URL and return its components.
    
    Args:
        url (str): The URL to parse
        
    Returns:
        dict: A dictionary containing the URL components:
            - scheme: URL scheme (e.g., 'http', 'https')
            - netloc: Network location (e.g., 'example.com')
            - path: URL path (e.g., '/images/photo.jpg')
            - params: Parameters (semicolon-separated)
            - query: Query string
            - fragment: Fragment identifier
    """
    parsed = urlparse(url)
    return {
        'scheme': parsed.scheme,
        'netloc': parsed.netloc,
        'path': parsed.path,
        'params': parsed.params,
        'query': parsed.query,
        'fragment': parsed.fragment
    }

def is_trusted_domain(url):
    """
    Check if a URL's domain is in the list of trusted domains.
    
    Args:
        url (str): The URL to check
        
    Returns:
        bool: True if the domain is trusted, False otherwise
    """
    parsed = urlparse(url)
    return parsed.netloc in TRUSTED_IMAGE_HOSTS

def fetch_image_url(url):
    """
    Verify that an image URL is from a trusted domain and uses an allowed scheme.
    
    Args:
        url (str): The URL to verify
        
    Returns:
        str: The verified URL if valid
        
    Raises:
        ValueError: If the domain is not trusted or the scheme is unrecognized
    """
    parsed = urlparse(url)
    
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"Unrecognized scheme: {parsed.scheme}")
    
    if parsed.netloc not in TRUSTED_IMAGE_HOSTS:
        raise ValueError(f"Untrusted domain: {parsed.netloc}")
    
    return url
