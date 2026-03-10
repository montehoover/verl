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
    Checks an RSS feed URL to ensure it comes from predefined trusted providers
    and does not point to an internal network address.

    Args:
        source_url: The RSS feed URL to check.

    Returns:
        The checked feed URL if valid and trusted.

    Raises:
        ValueError: If the URL is not from a trusted provider, uses an invalid
                    scheme, or points to an internal network address.
    """
    try:
        parsed_url = urlparse(source_url)
    except Exception as e:
        raise ValueError(f"Invalid URL format: {source_url}. Error: {e}")

    if not parsed_url.scheme or parsed_url.scheme.lower() not in ALLOWED_SCHEMES:
        raise ValueError(
            f"Invalid URL scheme: '{parsed_url.scheme}'. "
            f"Allowed schemes are: {', '.join(ALLOWED_SCHEMES)}."
        )

    hostname = parsed_url.hostname
    if not hostname:
        raise ValueError(f"Could not extract hostname from URL: {source_url}")

    # Check against internal IP patterns
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname, re.IGNORECASE):
            raise ValueError(
                f"Access to internal network address detected: {hostname}. "
                f"URL '{source_url}' is not allowed."
            )

    # Check against trusted providers
    if hostname.lower() not in TRUSTED_RSS_PROVIDERS:
        raise ValueError(
            f"URL '{source_url}' is not from a trusted provider. "
            f"Hostname '{hostname}' is not in the trusted list."
        )

    return source_url
