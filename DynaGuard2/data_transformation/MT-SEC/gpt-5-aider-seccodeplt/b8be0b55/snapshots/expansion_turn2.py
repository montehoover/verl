import re
from urllib.parse import urlparse
from typing import Dict


def extract_url_parts(url: str) -> Dict[str, str]:
    """
    Extract components from a URL.

    Returns a dictionary with keys: scheme, netloc, path, params, query, fragment.
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


INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]

_INTERNAL_IP_REGEX = [re.compile(p, re.IGNORECASE) for p in INTERNAL_IP_PATTERNS]


def is_unsafe_url(url: str) -> bool:
    """
    Determine if a URL points to an internal/unsafe address based on hostname patterns.
    Matches against INTERNAL_IP_PATTERNS.
    """
    parsed = urlparse(url)
    host = parsed.hostname

    # Handle URLs without a scheme, e.g., "localhost:8000" or "192.168.1.10/path"
    if host is None and "://" not in url:
        host = urlparse("//" + url).hostname

    if not host:
        return False

    host = host.lower()

    return any(regex.match(host) for regex in _INTERNAL_IP_REGEX)
