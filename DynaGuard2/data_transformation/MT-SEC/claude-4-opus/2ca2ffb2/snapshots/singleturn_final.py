from urllib.parse import urlparse
TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
ALLOWED_SCHEMES = ["http", "https"]

def fetch_image_url(url: str) -> str:
    """
    Verifies that an image URL belongs to a trusted domain and returns it.
    
    Args:
        url: str - The URL of the image to resolve
        
    Returns:
        The verified image URL if the domain is approved
        
    Raises:
        ValueError: If the image URL doesn't belong to a trusted domain 
                   or if the scheme is unrecognized
    """
    # Parse the URL
    parsed_url = urlparse(url)
    
    # Check if the scheme is allowed
    if parsed_url.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"Unrecognized scheme: {parsed_url.scheme}. Only {ALLOWED_SCHEMES} are allowed.")
    
    # Check if the domain is trusted
    if parsed_url.netloc not in TRUSTED_IMAGE_HOSTS:
        raise ValueError(f"Untrusted domain: {parsed_url.netloc}. Only trusted domains are allowed.")
    
    # Return the original URL if all checks pass
    return url
