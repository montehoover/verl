import re
from urllib.parse import urlparse

INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]

TRUSTED_RSS_PROVIDERS = ["rss.trustedsource.com", "feeds.provider.com", "newsfeed.safesite.org"]

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

def sanitize_feed_url(feed_url):
    """
    Ensures a feed URL is from a trusted provider and not an internal URL.

    Args:
        feed_url: The URL of the feed to sanitize.

    Returns:
        The sanitized URL if valid.

    Raises:
        ValueError: If the URL is not trusted or attempts to access a restricted network.
    """
    parsed_url = urlparse(feed_url)
    netloc = parsed_url.netloc

    if not netloc:
        raise ValueError("Invalid URL: No network location specified.")

    # Remove port if present for trusted provider check
    domain_to_check = netloc.split(":")[0]

    if domain_to_check not in TRUSTED_RSS_PROVIDERS:
        raise ValueError(f"URL provider '{domain_to_check}' is not trusted.")

    if is_internal_url(feed_url):
        raise ValueError("URL points to an internal or restricted network.")

    return feed_url
