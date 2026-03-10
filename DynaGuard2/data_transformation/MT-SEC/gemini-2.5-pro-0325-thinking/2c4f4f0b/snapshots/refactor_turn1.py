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

def check_rss_source(source_url: str) -> str:
    """
    Checks if an RSS feed URL is from a trusted provider and not an internal network address.

    Args:
        source_url: The RSS feed URL to check.

    Returns:
        The checked feed URL if valid and trusted.

    Raises:
        ValueError: If the URL is not from a trusted source, uses an invalid scheme,
                    or attempts to access an internal network address.
    """
    try:
        parsed_url = urlparse(source_url)
    except Exception as e:
        raise ValueError(f"Invalid URL format: {source_url}. Error: {e}")

    if not parsed_url.scheme or not parsed_url.hostname:
        raise ValueError(f"Invalid URL structure: {source_url}. Scheme and hostname are required.")

    if parsed_url.scheme.lower() not in ALLOWED_SCHEMES:
        raise ValueError(
            f"Invalid URL scheme: '{parsed_url.scheme}'. "
            f"Only {', '.join(ALLOWED_SCHEMES)} are allowed."
        )

    hostname = parsed_url.hostname
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            raise ValueError(
                f"Access to internal network address '{hostname}' is forbidden."
            )

    if hostname not in TRUSTED_RSS_PROVIDERS:
        raise ValueError(
            f"URL '{source_url}' is not from a trusted RSS provider. "
            f"Allowed providers are: {', '.join(TRUSTED_RSS_PROVIDERS)}"
        )

    return source_url
