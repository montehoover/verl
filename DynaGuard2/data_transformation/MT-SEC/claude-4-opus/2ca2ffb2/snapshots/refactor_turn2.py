from urllib.parse import urlparse
TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
ALLOWED_SCHEMES = ["http", "https"]

def validate_scheme(scheme: str) -> None:
    if scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"Unrecognized scheme: {scheme}")

def validate_domain(domain: str) -> None:
    if domain not in TRUSTED_IMAGE_HOSTS:
        raise ValueError(f"URL domain {domain} is not in the list of trusted domains")

def fetch_image_url(url: str) -> str:
    parsed_url = urlparse(url)
    
    validate_scheme(parsed_url.scheme)
    validate_domain(parsed_url.netloc)
    
    return url
