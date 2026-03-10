from urllib.parse import urlparse
from typing import Dict, List, Set

TRUSTED_IMAGE_HOSTS: List[str] = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
_TRUSTED_HOSTS_SET: Set[str] = {h.lower().rstrip('.') for h in TRUSTED_IMAGE_HOSTS}

def parse_image_url(url: str) -> Dict[str, str]:
    """
    Parse a URL and return its components as a dictionary.

    Keys: scheme, netloc, path, params, query, fragment
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

def is_trusted_domain(url: str) -> bool:
    """
    Return True if the URL's hostname is in the TRUSTED_IMAGE_HOSTS list.
    """
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().rstrip(".")
    if not host:
        return False
    return host in _TRUSTED_HOSTS_SET
