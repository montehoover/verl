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
        ValueError: If the URL is not from an approved provider, uses an
                    invalid scheme, or attempts to access restricted internal networks.
    """
    try:
        parsed_url = urlparse(rss_feed_url)
    except Exception as e:
        raise ValueError(f"Invalid URL format: {rss_feed_url}. Error: {e}")

    # Check scheme
    if parsed_url.scheme not in ALLOWED_SCHEMES:
        raise ValueError(
            f"Invalid URL scheme: '{parsed_url.scheme}'. "
            f"Allowed schemes are: {', '.join(ALLOWED_SCHEMES)}."
        )

    hostname = parsed_url.hostname
    if not hostname:
        raise ValueError("URL must contain a hostname.")

    # Check if hostname is an internal IP
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            raise ValueError(
                f"Access to internal or restricted network address '{hostname}' is not allowed."
            )

    # Check if provider is trusted
    if hostname not in TRUSTED_RSS_PROVIDERS:
        raise ValueError(
            f"URL provider '{hostname}' is not in the list of trusted providers. "
            f"Trusted providers are: {', '.join(TRUSTED_RSS_PROVIDERS)}."
        )

    return rss_feed_url
