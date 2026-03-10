from urllib.parse import urlparse
TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
ALLOWED_SCHEMES = ["http", "https"]

def is_valid_scheme(scheme: str) -> bool:
    return scheme in ALLOWED_SCHEMES

def is_trusted_domain(domain: str) -> bool:
    return domain in TRUSTED_IMAGE_HOSTS

def verify_image_url(img_url: str) -> str:
    parsed_url = urlparse(img_url)
    
    if not is_valid_scheme(parsed_url.scheme):
        raise ValueError(f"Unrecognized scheme: {parsed_url.scheme}")
    
    if not is_trusted_domain(parsed_url.netloc):
        raise ValueError(f"Untrusted domain: {parsed_url.netloc}")
    
    return img_url
