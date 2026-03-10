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
    Ensures that the URL points to a secure, whitelisted external API endpoint.
    Safeguards against possible SSRF attacks.
    
    Args:
        api_endpoint: The URL that the application is attempting to access
        
    Returns:
        The verified URL if it's allowed
        
    Raises:
        ValueError: If the URL targets an unapproved domain or any address within internal networks
    """
    # Parse the URL
    parsed_url = urlparse(api_endpoint)
    
    # Extract the hostname
    hostname = parsed_url.hostname
    if not hostname:
        raise ValueError(f"Invalid URL format: {api_endpoint}")
    
    # Check if the hostname matches any internal IP patterns
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            raise ValueError(f"Access to internal network addresses is not allowed: {api_endpoint}")
    
    # Check if the hostname is in the allowed API domains list
    if hostname not in ALLOWED_API_DOMAINS:
        raise ValueError(f"Access to domain '{hostname}' is not allowed. Only whitelisted domains are permitted.")
    
    # Return the verified URL
    return api_endpoint
