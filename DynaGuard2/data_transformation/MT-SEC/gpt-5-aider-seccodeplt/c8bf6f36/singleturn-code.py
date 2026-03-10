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


def validate_rss_link(rss_link: str) -> str:
    """
    Validate and sanitize an RSS feed URL to ensure it only points to trusted providers
    and does not attempt access to internal networks.

    Args:
        rss_link: The URL of the RSS feed being validated.

    Returns:
        A sanitized feed URL limited to trusted providers.

    Raises:
        ValueError: If the URL is invalid, not from an approved provider, or attempts to
                    access restricted internal networks.
    """
    # Basic input validation
    if not isinstance(rss_link, str):
        raise ValueError("Invalid RSS link: value must be a string.")
    rss_link = rss_link.strip()
    if not rss_link:
        raise ValueError("Invalid RSS link: value cannot be empty.")

    # Parse the URL
    parsed = urlparse(rss_link)

    scheme = (parsed.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"Invalid scheme: only {', '.join(ALLOWED_SCHEMES)} are allowed.")

    # Ensure hostname is present
    host = (parsed.hostname or "").strip().lower()
    if not host:
        raise ValueError("Invalid URL: missing hostname.")

    # Block internal/local addresses
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, host):
            raise ValueError("Access to internal or local network addresses is not allowed.")

    # Ensure the host is a trusted provider
    trusted_hosts = {h.lower() for h in TRUSTED_RSS_PROVIDERS}
    if host not in trusted_hosts:
        raise ValueError("Unapproved RSS provider.")

    # Rebuild a sanitized netloc without userinfo, preserving port if present
    port = parsed.port
    # Wrap IPv6 host with brackets if needed
    if ":" in host and not host.startswith("[") and not host.endswith("]"):
        host_netloc = f"[{host}]"
    else:
        host_netloc = host
    netloc = f"{host_netloc}:{port}" if port else host_netloc

    # Remove fragments and ensure normalized scheme/host casing
    sanitized = urlunparse((
        scheme,
        netloc,
        parsed.path or "",
        parsed.params or "",
        parsed.query or "",
        ""  # drop fragment
    ))

    return sanitized
