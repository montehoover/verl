from urllib.parse import urlparse

TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
ALLOWED_SCHEMES = ["http", "https"]

def verify_image_url(img_url: str) -> str:
    parsed = urlparse(img_url)
    
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"Invalid URL scheme: {parsed.scheme}")
    
    if parsed.netloc not in TRUSTED_IMAGE_HOSTS:
        raise ValueError(f"Untrusted image host: {parsed.netloc}")
    
    return img_url
