from urllib.parse import urlparse, urlunparse
import requests

ALLOWED_SCHEMES = ["http", "https"]

def verify_image_url(img_url: str) -> str:
    parsed = urlparse(img_url)
    
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"Invalid URL scheme: {parsed.scheme}")
    
    https_parsed = parsed._replace(scheme='https')
    https_url = urlunparse(https_parsed)
    
    try:
        response = requests.head(https_url, timeout=5)
        response.raise_for_status()
    except requests.RequestException:
        raise ValueError(f"URL is not reachable: {https_url}")
    
    return https_url
