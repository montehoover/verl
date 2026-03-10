from urllib.parse import urlparse
TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
ALLOWED_SCHEMES = ["http", "https"]

def is_scheme_allowed(scheme: str) -> bool:
    return scheme in ALLOWED_SCHEMES

def is_host_trusted(hostname: str) -> bool:
    return hostname in TRUSTED_IMAGE_HOSTS

def validate_scheme(scheme: str) -> None:
    if not is_scheme_allowed(scheme):
        raise ValueError(f"Unsupported scheme: {scheme}. Only {ALLOWED_SCHEMES} are allowed.")

def validate_host(hostname: str) -> None:
    if not is_host_trusted(hostname):
        raise ValueError(f"Untrusted domain: {hostname}. Only trusted hosts are allowed.")

def validate_image_url(img_url: str) -> str:
    parsed_url = urlparse(img_url)
    
    validate_scheme(parsed_url.scheme)
    validate_host(parsed_url.hostname)
    
    return img_url
