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
    Validates that a URL points to an external trusted API endpoint, preventing SSRF attacks.
    
    Args:
        api_url: str, the URL to be validated
        
    Returns:
        A validated URL that points to a trusted API
        
    Raises:
        ValueError: if the URL is not part of the allowed API domains or attempts to reach internal network addresses
    """
    # Parse the URL
    try:
        parsed_url = urlparse(api_url)
    except Exception as e:
        raise ValueError(f"Invalid URL format: {e}")
    
    # Extract the hostname
    hostname = parsed_url.hostname
    if not hostname:
        raise ValueError("URL must include a hostname")
    
    # Check if hostname matches any internal IP patterns
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            raise ValueError(f"URL points to internal network address: {hostname}")
    
    # Check if hostname is in allowed domains
    if hostname not in ALLOWED_API_DOMAINS:
        raise ValueError(f"URL domain '{hostname}' is not in the list of allowed API domains")
    
    # Ensure the URL uses HTTPS for security
    if parsed_url.scheme != 'https':
        raise ValueError("Only HTTPS URLs are allowed for API endpoints")
    
    return api_url
