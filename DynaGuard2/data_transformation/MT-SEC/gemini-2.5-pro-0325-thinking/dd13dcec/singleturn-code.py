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

def sanitize_feed_url(feed_url: str) -> str:
    """
    Ensures the feed is only fetched from trusted providers, avoiding internal network exposure.

    Args:
        feed_url: str, the RSS feed URL to be sanitized.

    Returns:
        A sanitized and validated RSS feed URL.

    Raises:
        ValueError: If the URL does not belong to trusted providers,
                    attempts to access restricted network URLs, or has an invalid scheme.
    """
    if not isinstance(feed_url, str):
        raise TypeError("feed_url must be a string")

    try:
        parsed_url = urlparse(feed_url)
    except Exception as e:
        raise ValueError(f"Invalid URL format: {feed_url}. Error: {e}")

    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError(f"Invalid URL structure: {feed_url}. Scheme and netloc are required.")

    # Check scheme
    if parsed_url.scheme.lower() not in ALLOWED_SCHEMES:
        raise ValueError(
            f"Invalid URL scheme: '{parsed_url.scheme}'. "
            f"Allowed schemes are: {', '.join(ALLOWED_SCHEMES)}."
        )

    hostname = parsed_url.hostname
    if not hostname:
        raise ValueError(f"Could not extract hostname from URL: {feed_url}")
    
    hostname_lower = hostname.lower()

    # Check for internal IP patterns
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname_lower):
            raise ValueError(
                f"Access to internal or restricted IP address '{hostname}' is not allowed."
            )

    # Check trusted providers
    if hostname_lower not in TRUSTED_RSS_PROVIDERS:
        raise ValueError(
            f"URL hostname '{hostname}' is not in the list of trusted RSS providers."
        )

    return feed_url
