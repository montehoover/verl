from urllib.parse import urlparse
TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
ALLOWED_SCHEMES = ["http", "https"]

def validate_image_url(img_url: str) -> str:
    """
    Validates an image URL and returns it if the domain is trusted.
    
    Args:
        img_url: str - The input image URL to validate
        
    Returns:
        str - The resolved valid URL if it belongs to the list of trusted domains
        
    Raises:
        ValueError - If the domain is not in the trusted list or the scheme is unsupported
    """
    # Parse the URL
    parsed_url = urlparse(img_url)
    
    # Extract scheme and domain
    scheme = parsed_url.scheme
    domain = parsed_url.netloc
    
    # Check if scheme is allowed
    if scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"Unsupported scheme '{scheme}'. Only {ALLOWED_SCHEMES} are allowed.")
    
    # Check if domain is trusted
    if domain not in TRUSTED_IMAGE_HOSTS:
        raise ValueError(f"Domain '{domain}' is not in the trusted hosts list.")
    
    # Return the full URL if all checks pass
    return img_url
