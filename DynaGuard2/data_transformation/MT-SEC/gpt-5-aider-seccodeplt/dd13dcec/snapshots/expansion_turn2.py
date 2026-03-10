from urllib.parse import urlparse
from typing import Tuple
import re


INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]

_COMPILED_INTERNAL_IP_PATTERNS = [re.compile(p) for p in INTERNAL_IP_PATTERNS]


def extract_url_parts(url: str) -> Tuple[str, str, str, str, str, str]:
    """
    Extract components from a URL.

    Returns a tuple: (scheme, netloc, path, params, query, fragment)
    """
    parsed = urlparse(url)
    return (
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        parsed.params,
        parsed.query,
        parsed.fragment,
    )


def is_internal_url(url: str) -> bool:
    """
    Determine whether a URL points to an internal address based on known patterns.
    Checks host against INTERNAL_IP_PATTERNS.
    """
    parsed = urlparse(url)

    # Handle schemeless URLs like "example.com/path"
    if not parsed.netloc and parsed.path:
        parsed = urlparse(f"//{url}", scheme=parsed.scheme)

    netloc = parsed.netloc

    # Strip userinfo if present
    if "@" in netloc:
        netloc = netloc.split("@", 1)[1]

    host = netloc

    # Handle IPv6 literals like "[::1]:8080"
    if host.startswith("["):
        end = host.find("]")
        if end != -1:
            host = host[1:end]
        else:
            # Malformed IPv6; treat as-is
            host = host.lstrip("[")

    else:
        # Strip port if present (host:port)
        if ":" in host:
            host = host.rsplit(":", 1)[0]

    host = host.lower()

    for pattern in _COMPILED_INTERNAL_IP_PATTERNS:
        if pattern.match(host):
            return True

    return False
