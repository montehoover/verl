import re
from urllib.parse import urlparse

ALLOWED_API_DOMAINS = ["api.trustedsource.com", "gateway.partnerapi.com"]
INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]

def is_internal_address(hostname):
    """
    Check if hostname matches any internal IP patterns.
    
    This function prevents SSRF attacks by detecting attempts to access
    internal network resources including localhost, loopback addresses,
    and private IP ranges.
    
    Args:
        hostname (str): The hostname or IP address to check
        
    Returns:
        bool: True if the hostname matches internal IP patterns, False otherwise
    """
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            return True
    return False

def is_allowed_domain(hostname):
    """
    Check if hostname is in the allowed domains list.
    
    This function maintains a whitelist of trusted external API domains
    to ensure requests are only made to approved endpoints.
    
    Args:
        hostname (str): The hostname to verify against the allowed list
        
    Returns:
        bool: True if the hostname is in the allowed domains list, False otherwise
    """
    return hostname in ALLOWED_API_DOMAINS

def enforce_api_security(api_endpoint):
    """
    Validate and secure an API endpoint URL against SSRF attacks.
    
    This function performs comprehensive security checks on a provided URL to ensure
    it points to a secure, external API endpoint. It enforces HTTPS protocol,
    blocks access to internal network addresses, and validates against a whitelist
    of approved API domains.
    
    Security checks performed:
    1. Validates URL has a proper scheme
    2. Enforces HTTPS-only connections
    3. Blocks internal network addresses (localhost, private IPs)
    4. Restricts access to whitelisted domains only
    
    Args:
        api_endpoint (str): The URL that needs to be verified
        
    Returns:
        str: The verified URL if all security checks pass
        
    Raises:
        ValueError: If the URL fails any security validation with detailed explanation:
            - Missing URL scheme
            - Non-HTTPS protocol
            - Invalid or missing hostname
            - Internal network address detected
            - Domain not in whitelist
    """
    # Parse the URL to extract components
    parsed_url = urlparse(api_endpoint)
    
    # Ensure the URL has a scheme and it's HTTPS
    if not parsed_url.scheme:
        raise ValueError(
            "URL must include a scheme (https://). "
            f"Provided URL '{api_endpoint}' is missing the protocol."
        )
    if parsed_url.scheme != "https":
        raise ValueError(
            f"Only HTTPS URLs are allowed for security. "
            f"Provided URL uses '{parsed_url.scheme}' protocol. "
            "Please use HTTPS to ensure encrypted communication."
        )
    
    # Get the hostname and validate it exists
    hostname = parsed_url.hostname
    if not hostname:
        raise ValueError(
            f"Invalid URL: no hostname found in '{api_endpoint}'. "
            "Please provide a complete URL with a valid domain or IP address."
        )
    
    # Check if the hostname matches any internal IP patterns
    if is_internal_address(hostname):
        raise ValueError(
            f"Access to internal network addresses is not allowed. "
            f"The hostname '{hostname}' appears to be an internal address. "
            "This restriction prevents Server-Side Request Forgery (SSRF) attacks."
        )
    
    # Check if the hostname is in the allowed domains list
    if not is_allowed_domain(hostname):
        raise ValueError(
            f"Domain not in allowed list: '{hostname}'. "
            f"Only the following domains are permitted: {', '.join(ALLOWED_API_DOMAINS)}. "
            "Contact your administrator to add new trusted domains."
        )
    
    # If all checks pass, return the verified URL
    return api_endpoint
