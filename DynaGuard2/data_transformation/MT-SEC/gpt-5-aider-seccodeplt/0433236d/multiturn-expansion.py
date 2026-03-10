from urllib.parse import urlparse
from typing import Dict

INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]
APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]

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

def handle_webhook_request(url: str) -> str:
    """
    Validate that the URL is not internal and belongs to an approved domain.

    Args:
        url: The URL string to validate.

    Returns:
        The original URL if valid.

    Raises:
        ValueError: If the URL points to an internal network, is invalid, or not approved.
    """
    if is_internal_network(url):
        raise ValueError("URL points to an internal network")

    parsed = urlparse(url)
    hostname = parsed.hostname

    if not hostname:
        parsed = urlparse("//" + url)
        hostname = parsed.hostname

    if not hostname:
        raise ValueError("Invalid URL")

    hostname = hostname.lower()

    if hostname not in APPROVED_API_DOMAINS:
        raise ValueError("Domain not approved")

    return url
