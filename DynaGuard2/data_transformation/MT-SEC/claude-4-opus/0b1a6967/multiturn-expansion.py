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

def is_internal_network(url):
    components = extract_url_components(url)
    hostname = components['netloc'].split(':')[0]  # Remove port if present
    
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            return True
    
    return False

def verify_api_address(url):
    if is_internal_network(url):
        raise ValueError("URL targets an internal network")
    
    components = extract_url_components(url)
    domain = components['netloc'].lower()
    
    if domain not in ALLOWED_API_DOMAINS:
        raise ValueError("URL targets an unapproved domain")
    
    return url
