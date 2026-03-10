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
        ValueError: If the URL is invalid, from an untrusted source,
                    or targets an internal network address.
    """
    try:
        parsed_url = urlparse(url)
    except Exception as e:
        raise ValueError(f"Invalid URL format: {url}. Error: {e}")

    if not parsed_url.scheme or not parsed_url.hostname:
        raise ValueError(f"Invalid URL structure: {url}. Scheme and hostname are required.")

    # 1. Validate scheme
    if parsed_url.scheme.lower() not in ALLOWED_SCHEMES:
        raise ValueError(
            f"Invalid URL scheme: '{parsed_url.scheme}'. "
            f"Allowed schemes are: {', '.join(ALLOWED_SCHEMES)}."
        )

    hostname = parsed_url.hostname

    # 2. Check against internal IP patterns
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname, re.IGNORECASE):
            raise ValueError(
                f"Access to internal network address is forbidden: {hostname}"
            )

    # 3. Check against trusted providers
    if hostname.lower() not in TRUSTED_RSS_PROVIDERS:
        raise ValueError(
            f"URL provider not trusted: {hostname}. "
            f"Allowed providers are: {', '.join(TRUSTED_RSS_PROVIDERS)}."
        )

    return url
