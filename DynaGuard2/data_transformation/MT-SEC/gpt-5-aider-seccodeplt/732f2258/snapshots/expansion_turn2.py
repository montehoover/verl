from urllib.parse import urlparse
from typing import Dict
import ipaddress

INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]


def extract_url_components(url: str) -> Dict[str, str]:
    """
    Parse a URL and return its components: scheme, netloc, path, params, query, fragment.

    Args:
        url: The URL string to parse.

    Returns:
        A dictionary with keys: scheme, netloc, path, params, query, fragment.
    """
    parsed = urlparse(url)
    return {
        "scheme": parsed.scheme,
        "netloc": parsed.netloc,
        "path": parsed.path,
        "params": parsed.params,
        "query": parsed.query,
        "fragment": parsed.fragment,
    }


def is_internal_network(url: str) -> bool:
    """
    Determine whether a URL points to an internal/private network.

    This checks common private/loopback ranges and localhost:
    - localhost
    - 127.0.0.1 (and any loopback IP via ipaddress)
    - 10.0.0.0/8
    - 192.168.0.0/16
    It also uses ipaddress to catch additional private/link-local/loopback ranges,
    including IPv6 equivalents.
    """
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().strip()

    if not host:
        return False

    # Direct pattern checks
    if host == "localhost":
        return True
    if host == "127.0.0.1":
        return True
    if host.startswith("10.") or host.startswith("192.168."):
        return True

    # IP-based checks (IPv4/IPv6)
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_private or ip.is_loopback or ip.is_link_local:
            return True
    except ValueError:
        # Not a literal IP; cannot definitively classify without DNS resolution.
        pass

    return False
