from urllib.parse import urlparse
from typing import Tuple, List
import re

INTERNAL_IP_PATTERNS: List[str] = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]

_INTERNAL_IP_REGEXES = [re.compile(p) for p in INTERNAL_IP_PATTERNS]

def extract_url_components(url: str) -> Tuple[str, str, str, str, str, str]:
    """
    Return (scheme, netloc, path, params, query, fragment) for the given URL.
    """
    parsed = urlparse(url)
    return parsed.scheme, parsed.netloc, parsed.path, parsed.params, parsed.query, parsed.fragment

def is_internal_network(url: str) -> bool:
    """
    Return True if the URL host matches a known internal network pattern.
    """
    parsed = urlparse(url)
    host = parsed.hostname

    # Handle schemeless URLs like "localhost:8000" or "192.168.1.1"
    if not host:
        parsed2 = urlparse("//" + url)
        host = parsed2.hostname or url

    if not host:
        return False

    host = host.strip().lower()

    for pattern in _INTERNAL_IP_REGEXES:
        if pattern.match(host):
            return True
    return False
