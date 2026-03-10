import re
from urllib.parse import urlparse

ALLOWED_API_DOMAINS = ["api.trustedsource.com", "gateway.partnerapi.com"]
INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]

def verify_api_address(endpoint_address: str) -> str:
    # Parse the URL
    parsed_url = urlparse(endpoint_address)
    
    # Check if scheme is https for security
    if parsed_url.scheme != 'https':
        raise ValueError("Only HTTPS URLs are allowed for secure API access")
    
    # Extract hostname
    hostname = parsed_url.hostname
    if not hostname:
        raise ValueError("Invalid URL: no hostname found")
    
    # Check against internal IP patterns
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            raise ValueError(f"Access to internal network address '{hostname}' is not allowed")
    
    # Check if domain is in allowed list
    if hostname not in ALLOWED_API_DOMAINS:
        raise ValueError(f"Domain '{hostname}' is not in the list of allowed API domains")
    
    # Return the verified URL
    return endpoint_address
