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
    Validate an RSS feed URL to ensure it originates from a trusted provider
    and does not target internal network addresses.

    Args:
        url: The RSS feed URL to validate.

    Returns:
        The original URL if it is valid and trusted.

    Raises:
        ValueError: If the URL is empty/invalid, uses a disallowed scheme,
                    does not match a trusted provider, or targets an internal network address.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("URL must be a non-empty string.")

    parsed = urlparse(url)

    scheme = (parsed.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        raise ValueError("URL scheme is not allowed. Allowed schemes are http and https.")

    # parsed.hostname excludes credentials and brackets, and is already lowercased by us
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL must include a valid hostname.")

    host = hostname.lower().strip(".")

    # Block internal addresses (localhost, loopback, and common private ranges)
    for pattern in INTERNAL_IP_PATTERNS:
        if re.fullmatch(pattern, host):
            raise ValueError("Access to internal network addresses is not allowed.")

    # Enforce exact match against trusted providers
    trusted_hosts = {h.lower() for h in TRUSTED_RSS_PROVIDERS}
    if host not in trusted_hosts:
        raise ValueError("URL host is not from a trusted provider.")

    return url
