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

def validate_rss_link(rss_link: str) -> str:
    """
    Ensures that the URL points only to trusted providers,
    thereby preventing access to insecure or internal networks.

    Args:
        rss_link: str, The URL of the RSS feed being validated.

    Returns:
        A properly sanitized feed URL limited to trusted providers.

    Raises:
        ValueError: If the input URL is not from an approved provider,
                    uses an invalid scheme, or if it attempts access
                    to restricted internal networks.
    """
    try:
        parsed_url = urlparse(rss_link)
    except Exception as e:
        raise ValueError(f"Invalid URL format: {rss_link}. Error: {e}")

    if not parsed_url.scheme or parsed_url.scheme.lower() not in ALLOWED_SCHEMES:
        raise ValueError(
            f"Invalid URL scheme: '{parsed_url.scheme}'. "
            f"Allowed schemes are: {', '.join(ALLOWED_SCHEMES)}."
        )

    hostname = parsed_url.hostname
    if not hostname:
        raise ValueError(f"URL must contain a hostname: {rss_link}")

    # Check against internal IP patterns
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            raise ValueError(
                f"Access to internal or restricted network address '{hostname}' is not allowed."
            )

    # Check if the hostname is in the trusted providers list
    if hostname not in TRUSTED_RSS_PROVIDERS:
        raise ValueError(
            f"URL hostname '{hostname}' is not a trusted RSS provider. "
            f"Allowed providers are: {', '.join(TRUSTED_RSS_PROVIDERS)}."
        )

    return rss_link
