from typing import Any, Dict
from urllib.parse import urlparse
import ipaddress


def _is_public_http_url(parts) -> bool:
    # Scheme must be http or https
    if parts.scheme not in ("http", "https"):
        return False

    # Must have a hostname
    host = parts.hostname
    if not host:
        return False

    # Reject obvious local hostnames
    lowered = host.lower()
    if lowered in {"localhost", "localhost.localdomain"}:
        return False

    # Handle potential IPv6 zone identifiers (e.g., fe80::1%eth0)
    host_for_ip_check = lowered.split("%", 1)[0]

    # If it's an IP literal, ensure it's not private/loopback/link-local/etc.
    try:
        ip = ipaddress.ip_address(host_for_ip_check)
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
        # Not an IP literal; allow domain names other than localhost
        pass

    return True


def extract_url_parts(url: str) -> Dict[str, Any]:
    """
    Parse a URL string and return its components in a dictionary,
    including a boolean 'is_valid' indicating whether the URL:
      - uses http or https
      - does not point to localhost or private/internal IP ranges
    """
    if not isinstance(url, str):
        raise TypeError("url must be a string")

    parts = urlparse(url)

    result: Dict[str, Any] = {
        "scheme": parts.scheme or None,
        "username": parts.username,
        "password": parts.password,
        "hostname": parts.hostname,
        "port": parts.port,
        "path": parts.path,
        "params": parts.params,
        "query": parts.query,
        "fragment": parts.fragment,
    }

    result["is_valid"] = _is_public_http_url(parts)
    return result
