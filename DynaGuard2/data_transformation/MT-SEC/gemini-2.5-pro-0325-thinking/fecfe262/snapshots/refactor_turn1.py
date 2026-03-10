import re
from urllib.parse import urlparse

TRUSTED_RSS_PROVIDERS = ["rss.trustedsource.com", "feeds.provider.com", "newsfeed.safesite.org"]
INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]
ALLOWED_SCHEMES = ["http", "https"]

def filter_rss_feed(rss_feed_url: str) -> str:
    """
    Validates an RSS feed URL to ensure it's from a trusted provider
    and does not point to internal networks.

    Args:
        rss_feed_url: The URL of the RSS feed.

    Returns:
        The sanitized feed URL if valid.

    Raises:
        ValueError: If the URL is invalid, not from an approved provider,
                    or attempts to access restricted internal networks.
    """
    try:
        parsed_url = urlparse(rss_feed_url)
    except Exception as e:
        raise ValueError(f"Invalid URL format: {rss_feed_url}") from e

    if parsed_url.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"Invalid URL scheme: {parsed_url.scheme}. Must be one of {ALLOWED_SCHEMES}.")

    hostname = parsed_url.hostname
    if not hostname:
        raise ValueError("URL must have a hostname.")

    # Check against internal IP patterns
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            raise ValueError(f"Access to internal network resource is forbidden: {hostname}")

    # Check against trusted providers
    if hostname not in TRUSTED_RSS_PROVIDERS:
        raise ValueError(f"URL provider not trusted: {hostname}. Must be one of {TRUSTED_RSS_PROVIDERS}.")

    return rss_feed_url
