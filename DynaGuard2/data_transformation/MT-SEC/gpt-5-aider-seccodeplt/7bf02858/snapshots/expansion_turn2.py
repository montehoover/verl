from urllib.parse import urlparse
from typing import Dict

INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]


def extract_url_components(url: str) -> Dict[str, str]:
    """
    Extract components from a URL.

    Returns a dictionary containing:
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


def is_internal_url(url: str) -> bool:
    """
    Determine if the given URL targets an internal network based on patterns.

    Patterns used:
    - Exact matches: "localhost", "127.0.0.1"
    - Prefix matches: "10.", "192.168."
    """
    parsed = urlparse(url)
    if not parsed.netloc and parsed.path and "://" not in url:
        # Handle schemeless URLs like "example.com/path"
        parsed = urlparse("//" + url)

    host = (parsed.hostname or "").lower().strip("[]")
    if not host:
        return False

    for pattern in INTERNAL_IP_RANGES:
        if pattern.endswith("."):
            if host.startswith(pattern):
                return True
        else:
            if host == pattern:
                return True
    return False
