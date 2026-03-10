from urllib.parse import urlparse
TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
ALLOWED_SCHEMES = ["http", "https"]

def resolve_image_url(image_url: str) -> str:
    parsed_url = urlparse(image_url)
    
    if parsed_url.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"URL scheme must be one of {ALLOWED_SCHEMES}")
    
    if parsed_url.hostname not in TRUSTED_IMAGE_HOSTS:
        raise ValueError(f"Image host '{parsed_url.hostname}' is not in the trusted list")
    
    return image_url
