from urllib.parse import urlparse

ALLOWED_SCHEMES = ["http", "https"]
TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]

def get_url_components(url):
    """
    Parse an image URL and return its fundamental components.
    
    Args:
        url (str): The image URL to parse
        
    Returns:
        dict: A dictionary containing the URL components:
            - scheme: The protocol (e.g., 'http', 'https')
            - netloc: The network location (domain and port)
            - path: The path to the resource
            - params: Parameters (semicolon-separated)
            - query: Query string (after the ?)
            - fragment: Fragment identifier (after the #)
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

def is_secure_scheme(url):
    """
    Check if a URL uses an allowed secure scheme.
    
    Args:
        url (str): The URL to check
        
    Returns:
        bool: True if the URL uses an allowed scheme, False otherwise
    """
    parsed = urlparse(url)
    return parsed.scheme in ALLOWED_SCHEMES

def validate_image_url(url):
    """
    Validate an image URL by checking if it uses a secure scheme and comes from a trusted domain.
    
    Args:
        url (str): The image URL to validate
        
    Returns:
        str: The validated URL
        
    Raises:
        ValueError: If the URL scheme is not supported or the domain is not trusted
    """
    parsed = urlparse(url)
    
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"Unsupported scheme: {parsed.scheme}")
    
    if parsed.netloc not in TRUSTED_IMAGE_HOSTS:
        raise ValueError(f"Untrusted domain: {parsed.netloc}")
    
    return url
