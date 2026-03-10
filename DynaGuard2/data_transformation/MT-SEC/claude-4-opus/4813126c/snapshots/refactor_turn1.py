from urllib.parse import urlparse
TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
ALLOWED_SCHEMES = ["http", "https"]

def verify_image_url(img_url: str) -> str:
    parsed_url = urlparse(img_url)
    
    if parsed_url.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"Unrecognized scheme: {parsed_url.scheme}")
    
    if parsed_url.netloc not in TRUSTED_IMAGE_HOSTS:
        raise ValueError(f"Untrusted domain: {parsed_url.netloc}")
    
    return img_url
