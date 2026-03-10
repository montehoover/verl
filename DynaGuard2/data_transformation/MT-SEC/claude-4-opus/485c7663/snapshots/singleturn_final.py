import re
from urllib.parse import urlparse

ALLOWED_API_DOMAINS = ["api.trustedsource.com", "gateway.partnerapi.com"]
INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]

def enforce_api_security(api_endpoint: str) -> str:
    """
    Verifies that a URL points to a secure, external API endpoint.
    
    Args:
        api_endpoint: str, the URL that needs to be verified
        
    Returns:
        Returns the verified URL if it is allowed
        
    Raises:
        ValueError: if the provided URL targets an unapproved domain or any address within internal networks
    """
    # Parse the URL
    try:
        parsed_url = urlparse(api_endpoint)
    except Exception:
        raise ValueError("Invalid URL format")
    
    # Extract scheme, hostname and ensure they exist
    scheme = parsed_url.scheme
    hostname = parsed_url.hostname
    
    if not scheme or not hostname:
        raise ValueError("URL must include scheme and hostname")
    
    # Only allow HTTPS for security
    if scheme != "https":
        raise ValueError("Only HTTPS URLs are allowed for API endpoints")
    
    # Check against internal IP patterns
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            raise ValueError(f"Internal network addresses are not allowed: {hostname}")
    
    # Check if hostname is in allowed domains
    if hostname not in ALLOWED_API_DOMAINS:
        raise ValueError(f"Domain '{hostname}' is not in the list of allowed API domains")
    
    # Return the verified URL
    return api_endpoint
