import re
from urllib.parse import urlparse
from typing import Any, Dict, Optional
import ipaddress

ALLOWED_API_DOMAINS = ["api.trustedsource.com", "gateway.partnerapi.com"]
INTERNAL_IP_PATTERNS = [
    r"^localhost$",
    r"^127\.0\.0\.\d+$",
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",
    r"^192\.168\.\d{1,3}\.\d{1,3}$",
]


def extract_url_parts(url: str) -> Dict[str, Optional[Any]]:
    """
    Parse a URL string and return its components as a dictionary.

    Returned keys:
      - scheme
      - username
      - password
      - hostname
      - port
      - path
      - query
      - fragment
    """
    if not isinstance(url, str):
        raise TypeError("url must be a string")

    parsed = urlparse(url)

    # Handle schemeless URLs like "example.com/path"
    if not parsed.scheme and not parsed.netloc and parsed.path:
        parsed = urlparse("//" + url)

    return {
        "scheme": parsed.scheme or None,
        "username": parsed.username or None,
        "password": parsed.password or None,
        "hostname": parsed.hostname or None,
        "port": parsed.port,
        "path": parsed.path or None,
        "query": parsed.query or None,
        "fragment": parsed.fragment or None,
    }


def check_url_security(url: str) -> bool:
    """
    Return True if the URL is considered secure, otherwise False.

    A URL is "secure" if:
      - It uses the HTTPS scheme, and
      - Its host does not point to an internal address or special-use name.

    Internal/special-use detection rules:
      - Literal IPs that are private, loopback, link-local, multicast, reserved, or unspecified are rejected.
      - Hostnames like "localhost" (and subdomains) are rejected.
      - Single-label hostnames (no dots) are treated as internal.
      - Schemeless URLs are handled similarly to extract_url_parts.
    """
    if not isinstance(url, str):
        raise TypeError("url must be a string")

    parsed = urlparse(url)
    if not parsed.scheme and not parsed.netloc and parsed.path:
        parsed = urlparse("//" + url)

    scheme_secure = (parsed.scheme or "").lower() == "https"
    host = parsed.hostname

    if not scheme_secure or not host:
        return False

    host_l = host.lower()

    # Reject localhost and its subdomains
    if host_l == "localhost" or host_l.endswith(".localhost"):
        return False

    # Treat single-label hostnames (no dot) as internal (e.g., "intranet")
    if "." not in host_l:
        return False

    # Check if host is a literal IP and whether it is internal/special-use
    try:
        ip = ipaddress.ip_address(host_l)
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return False
    except ValueError:
        # Not a literal IP; assume publicly routable hostname
        pass

    return True


def validate_api_endpoint(api_url: str) -> str:
    """
    Validate that api_url points to an external, trusted API endpoint.

    Requirements:
      - Must be a non-empty string.
      - Must use HTTPS scheme.
      - Hostname must not be internal (localhost, loopback, private ranges).
      - Hostname must match an allowed API domain (exact or subdomain).
      - Must not include user credentials.

    Returns the original api_url if valid, otherwise raises ValueError.
    """
    if not isinstance(api_url, str) or not api_url.strip():
        raise ValueError("api_url must be a non-empty string")

    api_url = api_url.strip()
    parsed = urlparse(api_url)

    if (parsed.scheme or "").lower() != "https":
        raise ValueError("URL must use https scheme")

    if not parsed.netloc or not parsed.hostname:
        raise ValueError("URL must include a valid hostname")

    if parsed.username or parsed.password:
        raise ValueError("URL must not include user credentials")

    host = parsed.hostname.lower()

    # Reject hosts that match known internal patterns (IPv4 and localhost)
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, host):
            raise ValueError("URL host is internal or reserved")

    # Reject literal IPs that are internal/special-use (IPv4/IPv6)
    try:
        ip = ipaddress.ip_address(host)
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise ValueError("URL host is internal or reserved")
    except ValueError:
        # Not a literal IP; continue with hostname checks
        pass

    # Check against allowed API domains (exact match or subdomain)
    host_allowed = False
    for dom in ALLOWED_API_DOMAINS:
        d = dom.lower()
        if host == d or host.endswith("." + d):
            host_allowed = True
            break

    if not host_allowed:
        raise ValueError("URL host is not in the list of allowed API domains")

    return api_url
