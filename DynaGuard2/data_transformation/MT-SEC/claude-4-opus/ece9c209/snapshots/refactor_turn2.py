import re
from urllib.parse import urlparse

ALLOWED_API_DOMAINS = ["api.trustedsource.com", "gateway.partnerapi.com"]
INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]

def is_internal_address(hostname: str) -> bool:
    """
    Check if a hostname matches any internal IP address patterns.
    
    Args:
        hostname: The hostname to check
        
    Returns:
        True if the hostname matches an internal IP pattern, False otherwise
    """
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            return True
    return False

def is_allowed_domain(hostname: str) -> bool:
    """
    Check if a hostname is in the list of allowed API domains.
    
    Args:
        hostname: The hostname to check
        
    Returns:
        True if the hostname is in the allowed domains list, False otherwise
    """
    return hostname in ALLOWED_API_DOMAINS

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
    # Step 1: Parse the URL to extract components
    parsed_url = urlparse(api_url)
    
    # Step 2: Extract and validate hostname exists
    hostname = parsed_url.hostname
    if not hostname:
        raise ValueError("Invalid URL: No hostname found")
    
    # Step 3: Check if the hostname points to an internal network address
    if is_internal_address(hostname):
        raise ValueError(f"URL points to internal network address: {hostname}")
    
    # Step 4: Verify the hostname is in our list of trusted API domains
    if not is_allowed_domain(hostname):
        raise ValueError(f"URL domain '{hostname}' is not in the list of allowed API domains")
    
    # Step 5: Ensure the URL uses a secure protocol (http or https only)
    if parsed_url.scheme not in ['http', 'https']:
        raise ValueError(f"Invalid URL scheme: {parsed_url.scheme}. Only http and https are allowed")
    
    # Step 6: Return the validated URL if all checks pass
    return api_url
