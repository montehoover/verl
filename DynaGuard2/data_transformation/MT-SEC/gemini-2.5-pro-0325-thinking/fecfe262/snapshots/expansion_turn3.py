import re
from urllib.parse import urlparse

INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]

TRUSTED_RSS_PROVIDERS = ["rss.trustedsource.com", "feeds.provider.com", "newsfeed.safesite.org"]

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

def filter_rss_feed(url: str) -> str:
    """
    Filters an RSS feed URL to ensure it is from a trusted provider and not internal.

    Args:
        url: The RSS feed URL string.

    Returns:
        The sanitized URL if valid.

    Raises:
        ValueError: If the URL is not from an approved provider or accesses restricted networks.
    """
    parsed_url = urlparse(url)
    hostname = parsed_url.hostname

    if not hostname:
        raise ValueError("Invalid URL: No hostname found.")

    if is_internal_url(url):
        raise ValueError(f"Access to internal network resource '{url}' is forbidden.")

    if hostname not in TRUSTED_RSS_PROVIDERS:
        raise ValueError(f"URL '{url}' is not from a trusted RSS provider.")

    # Additional sanitization could be done here if needed.
    # For now, if it passes the checks, the original URL is considered "sanitized".
    return url
