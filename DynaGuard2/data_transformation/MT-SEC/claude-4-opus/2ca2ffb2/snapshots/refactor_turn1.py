from urllib.parse import urlparse
TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
ALLOWED_SCHEMES = ["http", "https"]

def fetch_image_url(url: str) -> str:
    parsed_url = urlparse(url)
    
    if parsed_url.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"Unrecognized scheme: {parsed_url.scheme}")
    
    if parsed_url.netloc not in TRUSTED_IMAGE_HOSTS:
        raise ValueError(f"URL domain {parsed_url.netloc} is not in the list of trusted domains")
    
    return url
