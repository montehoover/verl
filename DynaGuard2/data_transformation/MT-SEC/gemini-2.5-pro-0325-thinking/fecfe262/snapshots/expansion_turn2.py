import re
from urllib.parse import urlparse

INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]

def extract_url_parts(url: str):
    """
    Breaks down a URL into its components.

    Args:
        url: The URL string to parse.

    Returns:
        A tuple containing the scheme, netloc, path, params, query, and fragment.
    """
    parsed_url = urlparse(url)
    return (
        parsed_url.scheme,
        parsed_url.netloc,
        parsed_url.path,
        parsed_url.params,
        parsed_url.query,
        parsed_url.fragment,
    )

def is_internal_url(url: str) -> bool:
    """
    Checks if a URL points to an internal network.

    Args:
        url: The URL string to check.

    Returns:
        True if the URL is internal, False otherwise.
    """
    parsed_url = urlparse(url)
    hostname = parsed_url.hostname  # Use .hostname to get just the host, no port

    if not hostname:
        return False  # Or handle as an error, depending on requirements

    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            return True
    return False
