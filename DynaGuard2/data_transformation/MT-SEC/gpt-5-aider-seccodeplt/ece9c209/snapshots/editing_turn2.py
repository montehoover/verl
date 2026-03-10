from urllib.parse import urlparse
from typing import Any, Dict, Optional
import ipaddress


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
