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


def sanitize_feed_url(feed_url: str) -> str:
    """
    Sanitize and validate an RSS feed URL to ensure it is fetched only from trusted providers
    and does not target internal/private network addresses.

    Args:
        feed_url: The RSS feed URL to be sanitized.

    Returns:
        A sanitized and validated RSS feed URL.

    Raises:
        ValueError: If the URL is invalid, uses a disallowed scheme, targets an internal/private
                    address, or does not belong to a trusted provider.
    """
    if not isinstance(feed_url, str) or not feed_url.strip():
        raise ValueError("feed_url must be a non-empty string")

    parsed = urlparse(feed_url.strip())

    # Validate scheme
    scheme = (parsed.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"URL scheme must be one of {ALLOWED_SCHEMES}")

    # Extract and normalize host
    host = parsed.hostname  # This strips user:pass@ and brackets around IPv6
    if not host:
        raise ValueError("URL must include a valid hostname")
    host = host.lower().rstrip(".")  # Normalize and prevent absolute-dot bypass

    # Block internal/private/local targets based on host patterns
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, host):
            raise ValueError("Access to internal or private network addresses is not allowed")

    # Allow only exact matches to known trusted providers
    trusted_hosts = {h.lower() for h in TRUSTED_RSS_PROVIDERS}
    if host not in trusted_hosts:
        raise ValueError("URL host is not in the list of trusted RSS providers")

    # Reconstruct a sanitized URL (no userinfo; preserve port/path/query/fragment)
    # Default to "/" if no path provided
    path = parsed.path or "/"
    params = parsed.params
    query = parsed.query
    fragment = parsed.fragment

    # Rebuild netloc with host and optional port (no username/password)
    # parsed.port may raise ValueError if the port is invalid; treat as invalid URL
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("URL contains an invalid port") from exc

    netloc = host if port is None else f"{host}:{port}"

    sanitized_url = urlunparse((scheme, netloc, path, params, query, fragment))
    return sanitized_url
