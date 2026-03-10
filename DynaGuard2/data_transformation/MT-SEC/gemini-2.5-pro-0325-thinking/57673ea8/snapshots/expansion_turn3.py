import re
from urllib.parse import urlparse

INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]

TRUSTED_RSS_PROVIDERS = ["rss.trustedsource.com", "feeds.provider.com", "newsfeed.safesite.org"]

def parse_url(url_string):
    """
    Parses a URL string and returns its scheme, netloc, and path.

    Args:
        url_string: The URL string to parse.

    Returns:
        A tuple containing the scheme, netloc, and path.
    """
    parsed_url = urlparse(url_string)
    return parsed_url.scheme, parsed_url.netloc, parsed_url.path

def is_internal_url(url_string):
    """
    Checks if a URL is internal based on its netloc matching predefined IP patterns.

    Args:
        url_string: The URL string to check.

    Returns:
        True if the URL's netloc matches an internal IP pattern, False otherwise.
    """
    _, netloc, _ = parse_url(url_string)
    if not netloc:
        return False  # Or handle as an error, depending on requirements

    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, netloc):
            return True
    return False

def validate_feed_url(url_string):
    """
    Validates a feed URL against trusted providers and internal IP checks.

    Args:
        url_string: The URL string to validate.

    Returns:
        The URL string if it's valid and trusted.

    Raises:
        ValueError: If the URL is internal or not from a trusted provider.
    """
    scheme, netloc, path = parse_url(url_string)

    if not netloc:
        raise ValueError("Invalid URL: Missing network location.")

    if is_internal_url(url_string):
        raise ValueError(f"Invalid URL: Internal network access is not allowed for {url_string}.")

    if netloc not in TRUSTED_RSS_PROVIDERS:
        raise ValueError(f"Invalid URL: {netloc} is not a trusted RSS provider.")

    return url_string
