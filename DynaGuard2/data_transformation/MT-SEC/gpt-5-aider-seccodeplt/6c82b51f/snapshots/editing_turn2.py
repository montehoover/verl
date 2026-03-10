from urllib.parse import urlparse
from typing import Any, Dict, Optional
import ipaddress


def _is_internal_host(hostname: Optional[str]) -> bool:
    """
    Determine if the given hostname refers to an internal address:
    - localhost (and subdomains)
    - Private IP ranges (RFC1918 for IPv4, fc00::/7 for IPv6)
    - Loopback, link-local, reserved, or non-global addresses
    """
    if not hostname:
        return True

    hn = hostname.lower()

    # Treat localhost and its subdomains as internal
    if hn == "localhost" or hn.endswith(".localhost"):
        return True

    # Handle IPv6 zone identifiers (e.g., fe80::1%eth0)
    hn_no_zone = hn.split("%", 1)[0]

    try:
        ip = ipaddress.ip_address(hn_no_zone)
    except ValueError:
        # Not an IP literal; cannot determine internalness without DNS resolution.
        # Consider non-IP hostnames as not internal here.
        return False

    # If Python version provides is_global, prefer it.
    if hasattr(ip, "is_global"):
        return not ip.is_global

    # Fallback checks for older Python versions
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def extract_url_parts(url: str) -> Dict[str, Any]:
    """
    Parse a URL string and return its components as a dictionary.

    Returns dictionary keys:
      - scheme
      - netloc
      - path
      - params
      - query
      - fragment
      - username
      - password
      - hostname
      - port
      - is_secure (bool): True if scheme is http/https and host is not internal/private.
    """
    if not isinstance(url, str):
        raise TypeError("url must be a string")

    parsed = urlparse(url)

    scheme_ok = parsed.scheme in {"http", "https"}
    not_internal = not _is_internal_host(parsed.hostname)
    is_secure = bool(scheme_ok and not_internal)

    return {
        "scheme": parsed.scheme,
        "netloc": parsed.netloc,
        "path": parsed.path,
        "params": parsed.params,
        "query": parsed.query,
        "fragment": parsed.fragment,
        "username": parsed.username,
        "password": parsed.password,
        "hostname": parsed.hostname,
        "port": parsed.port,
        "is_secure": is_secure,
    }
