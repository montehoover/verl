import re
from urllib.parse import urlparse

INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]

def extract_url_components(url_string):
    """
    Extracts components from a given URL string.

    Args:
        url_string (str): The URL to parse.

    Returns:
        tuple: A tuple containing the scheme, netloc, path, params, 
               query, and fragment of the URL.
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

def is_unsafe_url(url_string):
    """
    Checks if a given URL matches any patterns for internal IP addresses.

    Args:
        url_string (str): The URL to check.

    Returns:
        bool: True if the URL is considered unsafe, False otherwise.
    """
    parsed_url = urlparse(url_string)
    hostname = parsed_url.hostname  # Use .hostname to get just the host, not port

    if not hostname:
        return False  # Or True, depending on how you want to handle URLs without hostnames

    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            return True
    return False
