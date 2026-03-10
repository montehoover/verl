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
    Validate and sanitize an RSS feed URL to ensure it:
    - Uses an allowed scheme (http or https)
    - Does not target restricted/internal network addresses
    - Belongs to trusted RSS providers (either the exact host or their subdomains)
    - Contains no userinfo credentials

    Args:
        feed_url: The RSS feed URL to be sanitized.

    Returns:
        A sanitized and validated RSS feed URL (scheme and hostname normalized).

    Raises:
        ValueError: If the URL is invalid, uses a disallowed scheme, includes credentials,
                    targets restricted network locations, or is not from a trusted provider.
    """
    if not isinstance(feed_url, str) or not feed_url.strip():
        raise ValueError("feed_url must be a non-empty string")

    parsed = urlparse(feed_url.strip())

    # Validate scheme
    scheme = (parsed.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        raise ValueError("URL scheme must be http or https")

    # Must include a network location
    if not parsed.netloc:
        raise ValueError("URL must include a network location (host)")

    # Disallow embedded credentials (userinfo)
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Credentials in URL are not allowed")

    # Normalize and validate hostname
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL must include a valid hostname")

    # Convert to IDNA (punycode) for safe comparison and normalization
    try:
        host_idna = hostname.encode("idna").decode("ascii")
    except Exception:
        raise ValueError("Invalid hostname encoding")

    # Strip any trailing dot for canonical comparison
    host_idna = host_idna.rstrip(".").lower()

    # Reject IPv6 literals outright (trusted list contains DNS names only)
    if ":" in host_idna:
        raise ValueError("IPv6 literal hosts are not allowed")

    # Reject restricted/internal network targets by hostname/IP pattern
    for pattern in INTERNAL_IP_PATTERNS:
        if re.fullmatch(pattern, host_idna):
            raise ValueError("URL targets a restricted internal network address")

    # Ensure host is a trusted provider (exact match or subdomain)
    trusted_hosts = [h.lower() for h in TRUSTED_RSS_PROVIDERS]
    is_trusted = any(
        host_idna == t or host_idna.endswith("." + t)
        for t in trusted_hosts
    )
    if not is_trusted:
        raise ValueError("URL is not from a trusted RSS provider")

    # Validate and preserve port if present
    try:
        port = parsed.port  # accessing .port validates that it's a valid integer if present
    except ValueError:
        raise ValueError("Invalid port in URL")

    # Reconstruct a sanitized URL (drop fragment, normalize scheme/host)
    netloc = host_idna
    if port:
        netloc = f"{netloc}:{port}"

    sanitized = urlunparse((
        scheme,
        netloc,
        parsed.path or "/",  # ensure at least root path
        parsed.params,
        parsed.query,
        ""  # drop fragment
    ))

    return sanitized
