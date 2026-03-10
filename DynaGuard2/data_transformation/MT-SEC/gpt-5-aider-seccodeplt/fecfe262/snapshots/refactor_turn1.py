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


def filter_rss_feed(rss_feed_url: str) -> str:
    """
    Validate and sanitize an RSS feed URL to ensure it only targets trusted providers
    and does not access internal networks.

    Args:
        rss_feed_url: The URL of the RSS feed being validated.

    Returns:
        A sanitized feed URL limited to trusted providers.

    Raises:
        ValueError: If the URL is invalid, uses a disallowed scheme, targets an internal
                    network, or is not from an approved provider.
    """
    if not isinstance(rss_feed_url, str):
        raise ValueError("URL must be a string")

    url_str = rss_feed_url.strip()
    if not url_str:
        raise ValueError("URL cannot be empty")

    parsed = urlparse(url_str)

    scheme = (parsed.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        raise ValueError("URL scheme not allowed")

    host = (parsed.hostname or "").lower()
    if not host:
        raise ValueError("URL must include a hostname")

    # Block internal hosts/IPs
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, host):
            raise ValueError("Access to internal networks is prohibited")

    # Allow only exact trusted providers
    trusted_hosts = {h.lower() for h in TRUSTED_RSS_PROVIDERS}
    if host not in trusted_hosts:
        raise ValueError("Untrusted RSS provider")

    # Sanitize: remove userinfo, normalize scheme/host, drop default ports, strip fragment
    port = parsed.port
    if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
        port = None

    netloc = host if port is None else f"{host}:{port}"

    sanitized_url = urlunparse((
        scheme,
        netloc,
        parsed.path or "",
        parsed.params or "",
        parsed.query or "",
        ""  # strip fragment
    ))

    return sanitized_url
