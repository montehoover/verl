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

def clean_feed_link(rss_link: str) -> str:
    """
    Ensures an RSS feed URL comes from predefined trusted providers and does not point to an internal network.

    Args:
        rss_link: The RSS feed URL that requires sanitization.

    Returns:
        The sanitized feed URL if valid and trusted.

    Raises:
        ValueError: If the provided URL does not come from an authenticated source,
                    uses an disallowed scheme, or if access to an internal network address is detected.
    """
    try:
        parsed_url = urlparse(rss_link)
    except Exception as e:
        raise ValueError(f"Invalid URL format: {rss_link}. Error: {e}")

    if not parsed_url.scheme or parsed_url.scheme.lower() not in ALLOWED_SCHEMES:
        raise ValueError(f"Invalid URL scheme: {parsed_url.scheme}. Allowed schemes are: {', '.join(ALLOWED_SCHEMES)}")

    hostname = parsed_url.hostname
    if not hostname:
        raise ValueError(f"URL must contain a valid hostname: {rss_link}")

    # Check against internal IP patterns
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            raise ValueError(f"Access to internal network address is forbidden: {hostname}")

    # Check against trusted providers
    if hostname not in TRUSTED_RSS_PROVIDERS:
        raise ValueError(f"URL hostname '{hostname}' is not in the list of trusted providers.")

    return rss_link
