from urllib.parse import urlparse
import re

INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]

def extract_url_parts(url: str):
    """
    Parse the given URL and return its components.

    Returns a dictionary with keys:
    - scheme
    - netloc
    - path
    - params
    - query
    - fragment
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

def is_internal_network_url(url: str) -> bool:
    """
    Return True if the given URL points to an internal network host
    based on known internal IP/host patterns, otherwise False.
    """
    parsed = urlparse(url)
    host = parsed.hostname

    # Handle schemeless URLs (e.g., "example.com/path")
    if not host:
        parsed = urlparse("//" + url)
        host = parsed.hostname

    if not host:
        return False

    host = host.strip().lower().rstrip(".")

    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, host):
            return True

    return False
