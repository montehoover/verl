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
    Ensures a URL points to a secure, external API endpoint.

    Args:
        api_endpoint: The URL to verify.

    Returns:
        The verified URL if allowed.

    Raises:
        ValueError: If the URL is invalid, not HTTPS, targets an unapproved
                    domain, or targets an internal network address.
    """
    try:
        parsed_url = urlparse(api_endpoint)
    except Exception as e:
        raise ValueError(f"Invalid URL: {api_endpoint}. Error: {e}")

    if not parsed_url.scheme:
        raise ValueError(f"URL scheme is missing: {api_endpoint}")

    if parsed_url.scheme.lower() != "https":
        raise ValueError(f"URL must use HTTPS: {api_endpoint}")

    hostname = parsed_url.hostname
    if not hostname:
        raise ValueError(f"Hostname could not be determined from URL: {api_endpoint}")

    # Check against internal IP patterns
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            raise ValueError(f"Access to internal network address is forbidden: {hostname}")

    # Check against allowed domains
    if hostname not in ALLOWED_API_DOMAINS:
        # Additionally check if it's an IP address that wasn't caught by internal patterns
        # This helps prevent direct IP access unless explicitly allowed (which it isn't here)
        is_ip_address = all(c.isdigit() or c == '.' for c in hostname) # Basic IP check
        if is_ip_address: # More robust IP check could be re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", hostname)
             try:
                # Further check if it's a valid IP format and not something like 999.999.999.999
                import ipaddress
                ipaddress.ip_address(hostname)
             except ValueError:
                raise ValueError(f"Invalid IP address format in hostname: {hostname}")
             except ImportError: # Fallback if ipaddress module is not available
                pass # Basic check already done
             raise ValueError(f"Direct IP access is forbidden unless domain is allowed: {hostname}")

        raise ValueError(f"Domain not allowed: {hostname}. Allowed domains are: {', '.join(ALLOWED_API_DOMAINS)}")

    return api_endpoint
