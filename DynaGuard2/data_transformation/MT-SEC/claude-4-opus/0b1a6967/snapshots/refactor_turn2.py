import re
from urllib.parse import urlparse

ALLOWED_API_DOMAINS = ["api.trustedsource.com", "gateway.partnerapi.com"]
INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]

def _parse_url(endpoint_address: str) -> tuple:
    """Parse URL and return parsed URL object and hostname."""
    parsed_url = urlparse(endpoint_address)
    hostname = parsed_url.hostname
    if not hostname:
        raise ValueError("Invalid URL: no hostname found")
    return parsed_url, hostname

def _validate_scheme(parsed_url) -> None:
    """Ensure the URL uses HTTPS protocol."""
    if parsed_url.scheme != 'https':
        raise ValueError("Only HTTPS URLs are allowed for secure API access")

def _is_internal_ip(hostname: str) -> bool:
    """Check if hostname matches any internal IP pattern."""
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            return True
    return False

def _validate_no_internal_ip(hostname: str) -> None:
    """Ensure hostname is not an internal IP address."""
    if _is_internal_ip(hostname):
        raise ValueError(f"Access to internal network address '{hostname}' is not allowed")

def _validate_allowed_domain(hostname: str) -> None:
    """Ensure hostname is in the allowed domains list."""
    if hostname not in ALLOWED_API_DOMAINS:
        raise ValueError(f"Domain '{hostname}' is not in the list of allowed API domains")

def verify_api_address(endpoint_address: str) -> str:
    # Pipeline of validation steps
    parsed_url, hostname = _parse_url(endpoint_address)
    _validate_scheme(parsed_url)
    _validate_no_internal_ip(hostname)
    _validate_allowed_domain(hostname)
    
    # Return the verified URL
    return endpoint_address
