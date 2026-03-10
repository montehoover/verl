from urllib.parse import urlparse
TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
ALLOWED_SCHEMES = ["http", "https"]

def validate_image_url(img_url: str) -> str:
    parsed_url = urlparse(img_url)
    
    if parsed_url.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"Unsupported scheme: {parsed_url.scheme}. Only {ALLOWED_SCHEMES} are allowed.")
    
    if parsed_url.hostname not in TRUSTED_IMAGE_HOSTS:
        raise ValueError(f"Untrusted domain: {parsed_url.hostname}. Only trusted hosts are allowed.")
    
    return img_url
