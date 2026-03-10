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
    Validate that the RSS feed URL:
      - uses an allowed scheme (http/https),
      - does not point to internal network addresses,
      - and belongs to a trusted provider.

    Args:
        source_url: The RSS feed URL to validate.

    Returns:
        The original URL if valid and trusted.

    Raises:
        ValueError: If the URL is invalid, points to an internal address, or is not from a trusted provider.
    """
    if not isinstance(source_url, str) or not source_url.strip():
        raise ValueError("A non-empty RSS source URL string is required.")

    try:
        parsed = urlparse(source_url)
    except Exception as exc:
        raise ValueError("Invalid URL provided.") from exc

    scheme = (parsed.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        raise ValueError("URL scheme must be one of: http, https.")

    # Ensure hostname exists and strip any trailing dot for comparison safety.
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL must include a valid hostname.")
    host = hostname.strip(".").lower()

    # Disallow embedded credentials for safety.
    if parsed.username or parsed.password:
        raise ValueError("Credentials in URL are not allowed.")

    # Block internal network targets based on provided patterns.
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, host):
            raise ValueError("Access to internal network addresses is not allowed.")

    # Ensure the host is one of the trusted providers (exact match).
    trusted_hosts = {h.lower() for h in TRUSTED_RSS_PROVIDERS}
    if host not in trusted_hosts:
        raise ValueError("The RSS feed provider is not trusted.")

    return source_url
