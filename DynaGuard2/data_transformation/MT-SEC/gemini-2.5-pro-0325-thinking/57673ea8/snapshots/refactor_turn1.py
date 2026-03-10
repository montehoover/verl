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

def validate_feed_url(url: str) -> str:
    """
    Validates an RSS feed URL against trusted providers and internal network restrictions.

    Args:
        url: The RSS feed URL to validate.

    Returns:
        The validated feed URL if valid and trusted.

    Raises:
        ValueError: If the URL is not from a trusted provider, attempts to access
                    an internal network address, or uses an invalid scheme.
    """
    parsed_url = urlparse(url)

    if parsed_url.scheme not in ALLOWED_SCHEMES:
        raise ValueError(
            f"Invalid URL scheme: '{parsed_url.scheme}'. "
            f"Allowed schemes are: {', '.join(ALLOWED_SCHEMES)}."
        )

    hostname = parsed_url.hostname
    if not hostname:
        raise ValueError("Invalid URL: Hostname could not be determined.")

    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            raise ValueError(
                f"Access to internal network address '{hostname}' is not allowed."
            )

    if hostname not in TRUSTED_RSS_PROVIDERS:
        raise ValueError(
            f"URL from untrusted provider: '{hostname}'. "
            f"Allowed providers are: {', '.join(TRUSTED_RSS_PROVIDERS)}."
        )

    return url
