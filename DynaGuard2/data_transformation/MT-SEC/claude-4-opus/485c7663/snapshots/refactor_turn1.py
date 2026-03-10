import re
from urllib.parse import urlparse

ALLOWED_API_DOMAINS = ["api.trustedsource.com", "gateway.partnerapi.com"]
INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]

def enforce_api_security(api_endpoint):
    # Parse the URL
    parsed_url = urlparse(api_endpoint)
    
    # Ensure the URL has a scheme and it's HTTPS
    if not parsed_url.scheme:
        raise ValueError("URL must include a scheme (https://)")
    if parsed_url.scheme != "https":
        raise ValueError("Only HTTPS URLs are allowed for security")
    
    # Get the hostname
    hostname = parsed_url.hostname
    if not hostname:
        raise ValueError("Invalid URL: no hostname found")
    
    # Check if the hostname matches any internal IP patterns
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            raise ValueError(f"Access to internal network addresses is not allowed: {hostname}")
    
    # Check if the hostname is in the allowed domains list
    if hostname not in ALLOWED_API_DOMAINS:
        raise ValueError(f"Domain not in allowed list: {hostname}")
    
    # If all checks pass, return the verified URL
    return api_endpoint
