from urllib.parse import urlparse
from typing import Dict

INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]

def extract_url_components(url: str) -> Dict[str, str]:
    """
    Parse a URL and return its components.

    Args:
        url: The URL string to parse.

    Returns:
        A dict with keys: scheme, netloc, path, params, query, fragment.
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
    Determine if the given URL belongs to an internal network based on indicators.

    Args:
        url: The URL string to inspect.

    Returns:
        True if the URL hostname matches internal network indicators, else False.
    """
    parsed = urlparse(url)
    hostname = parsed.hostname

    if not hostname:
        # Attempt to parse URLs missing scheme by prefixing with //
        parsed = urlparse("//" + url)
        hostname = parsed.hostname

    if not hostname:
        return False

    hostname = hostname.lower()

    for indicator in INTERNAL_IP_RANGES:
        if indicator.endswith("."):
            if hostname.startswith(indicator):
                return True
        else:
            if hostname == indicator:
                return True

    return False
