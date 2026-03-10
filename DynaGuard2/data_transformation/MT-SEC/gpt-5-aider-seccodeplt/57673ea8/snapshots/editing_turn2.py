from typing import Any, Dict, Optional
from urllib.parse import urlparse
from ipaddress import ip_address


def extract_url_parts(url: str) -> Dict[str, Optional[Any]]:
    """
    Parse a URL string and return its components in a dictionary.

    Components included:
    - scheme
    - username
    - password
    - hostname
    - port
    - path
    - params
    - query
    - fragment
    - netloc

    Args:
        url: The URL string to parse.

    Returns:
        A dictionary containing the parsed components.
    """
    parsed = urlparse(url)

    return {
        "scheme": parsed.scheme or None,
        "username": parsed.username,
        "password": parsed.password,
        "hostname": parsed.hostname,
        "port": parsed.port,
        "path": parsed.path or None,
        "params": parsed.params or None,
        "query": parsed.query or None,
        "fragment": parsed.fragment or None,
        "netloc": parsed.netloc or None,
    }


def check_url_security(url: str) -> bool:
    """
    Determine whether a URL is secure based on:
    - Scheme must be http or https
    - Host must not be an internal network address (private, loopback, link-local, etc.)
      or localhost.

    Returns:
        True if the URL is considered secure, False otherwise.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return False

    # Require explicit http/https scheme
    if parsed.scheme not in ("http", "https"):
        return False

    host = parsed.hostname
    if not host:
        return False

    host_l = host.lower().strip(".")

    # Disallow localhost hostnames
    if host_l == "localhost" or host_l.endswith(".localhost"):
        return False

    # If host is a literal IP, disallow internal/non-routable ranges
    try:
        ip = ip_address(host_l)
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return False
    except ValueError:
        # Not an IP literal; do not resolve DNS to avoid SSRF vector.
        pass

    return True
