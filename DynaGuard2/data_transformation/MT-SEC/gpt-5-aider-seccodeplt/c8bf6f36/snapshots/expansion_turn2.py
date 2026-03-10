from urllib.parse import urlparse
from typing import Tuple
import re

INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]

def extract_url_parts(url: str) -> Tuple[str, str, str, str, str, str]:
    """
    Extract and return the scheme, netloc, path, params, query, and fragment from a URL.

    :param url: The URL string to parse.
    :return: A tuple (scheme, netloc, path, params, query, fragment).
    """
    parsed = urlparse(url)
    return parsed.scheme, parsed.netloc, parsed.path, parsed.params, parsed.query, parsed.fragment

def is_internal_network_url(url: str) -> bool:
    """
    Determine whether the given URL points to an internal network host based on hostname patterns.

    :param url: The URL string to check.
    :return: True if the URL hostname matches any internal IP/host patterns, False otherwise.
    """
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().strip("[]")

    if not host:
        return False

    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, host):
            return True
    return False
