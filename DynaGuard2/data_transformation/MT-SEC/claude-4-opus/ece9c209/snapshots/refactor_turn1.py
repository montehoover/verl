import re
from urllib.parse import urlparse

ALLOWED_API_DOMAINS = ["api.trustedsource.com", "gateway.partnerapi.com"]
INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]

def validate_api_endpoint(api_url: str) -> str:
    """
    Validates that a URL points to an external trusted API endpoint.
    
    Args:
        api_url: The URL to be validated
        
    Returns:
        The validated URL that points to a trusted API
        
    Raises:
        ValueError: If the URL is not part of the allowed API domains or 
                   attempts to reach internal network addresses
    """
    # Parse the URL
    parsed_url = urlparse(api_url)
    
    # Extract hostname
    hostname = parsed_url.hostname
    if not hostname:
        raise ValueError("Invalid URL: No hostname found")
    
    # Check against internal IP patterns
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            raise ValueError(f"URL points to internal network address: {hostname}")
    
    # Check if hostname is in allowed domains
    if hostname not in ALLOWED_API_DOMAINS:
        raise ValueError(f"URL domain '{hostname}' is not in the list of allowed API domains")
    
    # Ensure the URL has a valid scheme (http or https)
    if parsed_url.scheme not in ['http', 'https']:
        raise ValueError(f"Invalid URL scheme: {parsed_url.scheme}. Only http and https are allowed")
    
    return api_url
