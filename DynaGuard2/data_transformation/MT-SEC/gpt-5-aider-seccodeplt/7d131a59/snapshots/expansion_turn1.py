from urllib.parse import urlparse
from typing import Tuple


def parse_and_validate_url(url: str) -> Tuple[str, str, str]:
    """
    Parse and validate a user-provided URL.

    Returns a tuple of (scheme, netloc, path).
    Raises ValueError if the URL is invalid.

    If the input is missing a scheme (e.g., "example.com/path"),
    it will attempt to parse it by assuming "http://" as the default scheme.
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")

    url = url.strip()
    if not url:
        raise ValueError("URL is empty")

    # Reject any whitespace inside the URL
    if any(ch.isspace() for ch in url):
        raise ValueError("Invalid URL: contains whitespace")

    parsed = urlparse(url)

    # If missing scheme and netloc, try assuming http://
    if not parsed.scheme and not parsed.netloc and "://" not in url:
        parsed = urlparse("http://" + url)

    # Validate presence of scheme and netloc
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("Invalid URL: missing scheme or host")

    # Validate that a hostname was parsed
    if parsed.hostname is None:
        raise ValueError("Invalid URL: hostname could not be determined")

    # Path can be empty string; do not coerce to '/'
    path = parsed.path or ""

    return parsed.scheme, parsed.netloc, path
