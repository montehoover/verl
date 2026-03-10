import re
from urllib.parse import urlparse, urlunparse

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
    Validate and sanitize an RSS feed URL to ensure it originates from a trusted provider
    and does not target internal network addresses.

    Args:
        rss_link: The RSS feed URL to validate and sanitize.

    Returns:
        The sanitized feed URL if it is valid and trusted.

    Raises:
        ValueError: If the URL is missing/invalid, uses a disallowed scheme, targets an
                    internal network address, or does not belong to a trusted provider.
    """
    if not isinstance(rss_link, str):
        raise ValueError("RSS link must be a string.")
    rss_link = rss_link.strip()
    if not rss_link:
        raise ValueError("RSS link cannot be empty.")

    parsed = urlparse(rss_link)

    # Validate scheme
    scheme = (parsed.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        raise ValueError("Unsupported or missing URL scheme. Allowed: http, https.")

    # Extract normalized hostname (excludes credentials and port)
    host = (parsed.hostname or "").lower().strip()
    if not host:
        raise ValueError("URL must include a valid host.")

    # Block internal network hosts based on provided patterns
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, host):
            raise ValueError("Access to internal network addresses is not allowed.")

    # Enforce trusted providers (exact host match)
    if host not in TRUSTED_RSS_PROVIDERS:
        raise ValueError("Untrusted RSS provider.")

    # Rebuild a sanitized URL without userinfo, preserving optional explicit port
    netloc = host
    if parsed.port:
        netloc = f"{host}:{parsed.port}"

    path = parsed.path or "/"

    sanitized = urlunparse((
        scheme,
        netloc,
        path,
        parsed.params,
        parsed.query,
        parsed.fragment,
    ))
    return sanitized
