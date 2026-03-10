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
                    or attempts to access a restricted network area.
    """
    try:
        parsed_url = urlparse(rss_link)
    except Exception as e:
        raise ValueError(f"Invalid URL format: {rss_link}. Error: {e}")

    if not parsed_url.scheme or not parsed_url.hostname:
        raise ValueError(f"Invalid URL structure: {rss_link}. Scheme and hostname are required.")

    # Check scheme
    if parsed_url.scheme.lower() not in ALLOWED_SCHEMES:
        raise ValueError(
            f"Invalid URL scheme: '{parsed_url.scheme}'. "
            f"Allowed schemes are: {', '.join(ALLOWED_SCHEMES)}."
        )

    # Check hostname against trusted providers
    hostname = parsed_url.hostname.lower()
    if hostname not in TRUSTED_RSS_PROVIDERS:
        raise ValueError(
            f"URL hostname '{hostname}' is not a trusted provider. "
            f"Trusted providers are: {', '.join(TRUSTED_RSS_PROVIDERS)}."
        )

    # Check hostname against internal IP patterns
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            raise ValueError(
                f"Access to internal network address '{hostname}' is restricted."
            )

    return rss_link
