from urllib.parse import urlparse
import re

INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]

ALLOWED_API_DOMAINS = ["api.trustedsource.com", "gateway.partnerapi.com"]

def extract_url_components(url):
    parsed = urlparse(url)
    return {
        'scheme': parsed.scheme,
        'netloc': parsed.netloc,
        'path': parsed.path,
        'params': parsed.params,
        'query': parsed.query,
        'fragment': parsed.fragment
    }

def is_unsafe_url(url):
    components = extract_url_components(url)
    netloc = components['netloc']
    
    # Extract hostname from netloc (remove port if present)
    hostname = netloc.split(':')[0]
    
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            return True
    
    return False

def validate_api_endpoint(url):
    if is_unsafe_url(url):
        raise ValueError("URL points to internal network")
    
    components = extract_url_components(url)
    netloc = components['netloc']
    
    # Extract hostname from netloc (remove port if present)
    hostname = netloc.split(':')[0]
    
    if hostname not in ALLOWED_API_DOMAINS:
        raise ValueError("URL is not from allowed API domains")
    
    return url
