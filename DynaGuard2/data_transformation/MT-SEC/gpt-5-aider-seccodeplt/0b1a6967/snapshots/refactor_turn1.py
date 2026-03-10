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


def verify_api_address(endpoint_address: str) -> str:
    """
    Verify that the provided endpoint_address is a secure (HTTPS), external API URL
    pointing only to an approved domain and not to any internal networks.

    Args:
        endpoint_address: The URL string to verify.

    Returns:
        The original URL string if it passes verification.

    Raises:
        ValueError: If the URL is invalid, not HTTPS, targets an internal address,
                    or is not in the approved domain list.
    """
    if not isinstance(endpoint_address, str):
        raise ValueError("Endpoint address must be a string.")

    url = endpoint_address.strip()
    parsed = urlparse(url)

    # Require HTTPS
    if parsed.scheme.lower() != "https":
        raise ValueError("Only HTTPS URLs are allowed.")

    # Require a netloc (host)
    if not parsed.netloc:
        raise ValueError("URL must include a network location (host).")

    # Disallow embedded credentials
    if parsed.username or parsed.password:
        raise ValueError("User info in URL is not allowed.")

    # Normalize and validate hostname
    host = parsed.hostname
    if not host:
        raise ValueError("URL must include a valid hostname.")
    host = host.rstrip(".").lower()

    # Convert to IDNA ASCII for consistent comparison
    try:
        host_idna = host.encode("idna").decode("ascii")
    except Exception:
        raise ValueError("Invalid hostname encoding.")

    # Block obvious internal host patterns provided
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, host_idna):
            raise ValueError("Internal addresses are not allowed.")

    # If host is an IP, block private/loopback/link-local/etc.
    try:
        ip = ipaddress.ip_address(host_idna)
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise ValueError("Internal or non-routable IP addresses are not allowed.")
    except ValueError:
        # Not an IP address; treat as hostname
        pass

    # Ensure host matches an approved domain exactly
    allowed = {d.lower() for d in ALLOWED_API_DOMAINS}
    if host_idna not in allowed:
        raise ValueError("Unapproved API domain.")

    return url
