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
    Validate that the given URL:
    - Uses an allowed scheme (http/https)
    - Does not point to internal network addresses
    - Belongs to a trusted RSS provider (exact domain or its subdomain)

    Returns the original URL if valid, otherwise raises ValueError.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("A non-empty URL string is required.")

    url = url.strip()
    parsed = urlparse(url)

    scheme = (parsed.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        raise ValueError("URL scheme must be http or https.")

    host = (parsed.hostname or "").lower()
    if not host:
        raise ValueError("URL must include a valid host.")

    # Block obvious internal network hosts (by pattern)
    for pattern in INTERNAL_IP_PATTERNS:
        if re.fullmatch(pattern, host):
            raise ValueError("Access to internal network addresses is not allowed.")

    # Enforce trusted providers (exact match or subdomain of a trusted domain)
    trusted = False
    for provider in TRUSTED_RSS_PROVIDERS:
        p = provider.lower()
        if host == p or host.endswith("." + p):
            trusted = True
            break

    if not trusted:
        raise ValueError("URL is not from a trusted RSS provider.")

    return url
