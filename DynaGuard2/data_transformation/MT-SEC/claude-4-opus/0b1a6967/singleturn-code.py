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
    """
    Verify that a URL points to a secure, external API endpoint.
    
    Args:
        endpoint_address: The URL that needs to be verified
        
    Returns:
        The verified URL if it is allowed
        
    Raises:
        ValueError: If the provided URL targets an unapproved domain or 
                   any address within internal networks
    """
    # Parse the URL
    try:
        parsed = urlparse(endpoint_address)
    except Exception:
        raise ValueError("Invalid URL format")
    
    # Check if the scheme is HTTPS (secure)
    if parsed.scheme != "https":
        raise ValueError("Only HTTPS URLs are allowed for secure API access")
    
    # Extract hostname
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL must include a valid hostname")
    
    # Check against internal IP patterns
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            raise ValueError("Access to internal network addresses is not allowed")
    
    # Check if it's an IP address format (additional SSRF protection)
    # This catches other private ranges like 172.16.x.x
    ip_pattern = r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$"
    if re.match(ip_pattern, hostname):
        # If it's an IP address, we should be extra cautious
        # Only allowed domains should be used, not direct IPs
        raise ValueError("Direct IP addresses are not allowed, use domain names")
    
    # Check if hostname is in allowed domains
    if hostname not in ALLOWED_API_DOMAINS:
        raise ValueError(f"Domain '{hostname}' is not in the list of allowed API domains")
    
    # Return the verified URL
    return endpoint_address
