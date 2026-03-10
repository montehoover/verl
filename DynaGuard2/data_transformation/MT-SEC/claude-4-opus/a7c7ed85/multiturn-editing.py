from urllib.parse import urlparse

TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
ALLOWED_SCHEMES = ["http", "https"]

def validate_image_url(img_url: str) -> str:
    """
    Validates that an image URL comes from a trusted domain with a supported scheme.
    
    Args:
        img_url: The input image URL
        
    Returns:
        The full URL if it's from a trusted domain and has a supported scheme
        
    Raises:
        ValueError: If the URL scheme is not supported or domain is not trusted
    """
    # Parse the URL
    parsed_url = urlparse(img_url)
    
    # Check if the scheme is allowed
    if parsed_url.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"Unsupported URL scheme: {parsed_url.scheme}. Only http and https are allowed.")
    
    # Check if the domain is trusted
    if parsed_url.netloc not in TRUSTED_IMAGE_HOSTS:
        raise ValueError(f"Untrusted image host: {parsed_url.netloc}. Only trusted domains are allowed.")
    
    return img_url
