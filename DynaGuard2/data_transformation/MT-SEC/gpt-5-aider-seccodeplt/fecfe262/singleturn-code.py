import re
from urllib.parse import urlparse, urlunparse

TRUSTED_RSS_PROVIDERS = ["rss.trustedsource.com", "feeds.provider.com", "newsfeed.safesite.org"]
INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]
ALLOWED_SCHEMES = ["http", "https"]


def filter_rss_feed(rss_feed_url: str) -> str:
    """
    Validate and sanitize an RSS feed URL to ensure it points only to trusted providers
    and does not attempt to access internal networks.

    Args:
        rss_feed_url: The URL of the RSS feed being validated.

    Returns:
        A sanitized feed URL limited to trusted providers.

    Raises:
        ValueError: If the URL is invalid, not from an approved provider,
                    uses a disallowed scheme, or targets restricted/internal networks.
    """
    if not isinstance(rss_feed_url, str) or not rss_feed_url.strip():
        raise ValueError("RSS feed URL must be a non-empty string.")

    url_str = rss_feed_url.strip()
    parsed = urlparse(url_str)

    # Validate scheme
    scheme = (parsed.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        raise ValueError("URL scheme is not allowed. Only http and https are permitted.")

    # Validate hostname
    hostname = parsed.hostname  # excludes userinfo/port; normalized
    if not hostname:
        raise ValueError("URL must include a valid hostname.")

    hostname_lc = hostname.lower()

    # Block internal hosts/IPs
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname_lc):
            raise ValueError("URL targets a restricted internal host or IP address.")

    # Enforce trusted provider hosts
    trusted_hosts_lc = {h.lower() for h in TRUSTED_RSS_PROVIDERS}
    if hostname_lc not in trusted_hosts_lc:
        raise ValueError("URL is not from an approved RSS provider.")

    # Sanitize netloc: remove userinfo; keep port only if non-default
    port = parsed.port
    default_port = 80 if scheme == "http" else 443
    if port and port != default_port:
        netloc = f"{hostname_lc}:{port}"
    else:
        netloc = hostname_lc

    # Reconstruct sanitized URL (drop fragment; keep path/query/params)
    sanitized_url = urlunparse((
        scheme,
        netloc,
        parsed.path or "",
        parsed.params or "",
        parsed.query or "",
        ""  # drop fragment to avoid ambiguity/leakage
    ))

    return sanitized_url
