from urllib.parse import urlparse

TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
ALLOWED_SCHEMES = ["http", "https"]

def resolve_image_url(image_url: str) -> str:
    """
    Resolves an image URL after verifying it belongs to a trusted image hosting service.
    
    Args:
        image_url: str, the image URL to be resolved
        
    Returns:
        The resolved URL if it belongs to a trusted domain
        
    Raises:
        ValueError: if the image URL's domain is not in the trusted list
    """
    # Parse the URL
    parsed_url = urlparse(image_url)
    
    # Check if the scheme is allowed
    if parsed_url.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"URL scheme '{parsed_url.scheme}' is not allowed. Only {ALLOWED_SCHEMES} are permitted.")
    
    # Check if the hostname is in the trusted list
    if parsed_url.hostname not in TRUSTED_IMAGE_HOSTS:
        raise ValueError(f"Domain '{parsed_url.hostname}' is not in the trusted image hosts list.")
    
    # Return the resolved URL (in this case, the same URL since it's already valid)
    return image_url
