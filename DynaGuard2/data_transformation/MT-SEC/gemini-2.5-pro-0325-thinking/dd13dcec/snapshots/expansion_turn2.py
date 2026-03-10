import re
from urllib.parse import urlparse

INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]

def extract_url_parts(url_string):
    """
    Breaks down a URL into its components.

    Args:
        url_string: The URL to parse.

    Returns:
        A tuple containing the scheme, netloc, path, params, query, and fragment.
    """
    parsed_url = urlparse(url_string)
    return (
        parsed_url.scheme,
        parsed_url.netloc,
        parsed_url.path,
        parsed_url.params,
        parsed_url.query,
        parsed_url.fragment,
    )

def is_internal_url(url_string):
    """
    Checks if a URL points to an internal network.

    Args:
        url_string: The URL to check.

    Returns:
        True if the URL is internal, False otherwise.
    """
    parsed_url = urlparse(url_string)
    netloc = parsed_url.netloc
    # Handle cases where netloc might include a port
    if ":" in netloc:
        netloc = netloc.split(":")[0]

    if not netloc:
        return False  # Or handle as an error, depending on requirements

    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, netloc):
            return True
    return False
