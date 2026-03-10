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

def check_feed_url(rss_link: str) -> str:
    """
    Validates an RSS feed URL, ensuring it's from a trusted provider
    and does not access internal networks.

    Args:
        rss_link: The RSS feed URL to validate.

    Returns:
        The valid, sanitized RSS feed URL.

    Raises:
        ValueError: If the URL is invalid, not from a trusted provider,
                    or attempts to connect to a restricted network area.
    """
    try:
        parsed_url = urlparse(rss_link)
    except Exception as e:
        raise ValueError(f"Invalid URL: {rss_link}. Error: {e}")

    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError(f"Invalid URL structure: {rss_link}")

    if parsed_url.scheme.lower() not in ALLOWED_SCHEMES:
        raise ValueError(f"Invalid URL scheme: {parsed_url.scheme}. Allowed schemes are: {', '.join(ALLOWED_SCHEMES)}")

    hostname = parsed_url.hostname
    if hostname is None:
        raise ValueError(f"Could not determine hostname from URL: {rss_link}")

    if hostname.lower() not in TRUSTED_RSS_PROVIDERS:
        raise ValueError(f"URL provider not trusted: {hostname}. Must be one of: {', '.join(TRUSTED_RSS_PROVIDERS)}")

    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            raise ValueError(f"Access to internal or restricted network address is forbidden: {hostname}")

    return rss_link
