import re
import ipaddress
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
    Ensures the given API endpoint URL is a secure (HTTPS), external endpoint
    and not targeting internal networks. Returns the verified URL if allowed.

    Args:
        api_endpoint: str - The URL that needs to be verified.

    Returns:
        str: The verified URL if it is allowed.

    Raises:
        ValueError: If the URL is invalid, not HTTPS, targets an unapproved domain,
                    or any address within internal networks.
    """
    if not isinstance(api_endpoint, str) or not api_endpoint.strip():
        raise ValueError("A non-empty API endpoint URL string is required.")

    parsed = urlparse(api_endpoint)

    # Require HTTPS for secure access
    scheme = (parsed.scheme or "").lower()
    if scheme != "https":
        raise ValueError("Only HTTPS API endpoints are allowed.")

    # Ensure a hostname exists
    hostname = (parsed.hostname or "").strip().lower()
    if not hostname:
        raise ValueError("Invalid URL: hostname is missing.")

    # Normalize hostname (remove trailing dot which can appear on FQDNs)
    hostname = hostname.rstrip(".")

    # Block direct access to internal IPs via robust IP checks (IPv4/IPv6)
    try:
        ip = ipaddress.ip_address(hostname)
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
        ):
            raise ValueError("Access to internal or non-public IP addresses is not allowed.")
    except ValueError:
        # Not a literal IP address; continue with hostname checks
        pass

    # Block obvious internal hostnames (localhost, loopback and RFC1918 patterns)
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            raise ValueError("Access to internal network addresses is not allowed.")

    # Allow only explicitly approved external API domains (exact match)
    allowed = {d.lower() for d in ALLOWED_API_DOMAINS}
    if hostname not in allowed:
        raise ValueError("Unapproved API domain.")

    # If all checks pass, return the original URL (unchanged)
    return api_endpoint
