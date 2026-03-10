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


def _is_internal_host(host: str) -> bool:
    """
    Returns True if the host matches any restricted/internal network pattern.
    """
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, host):
            return True
    return False


def _is_trusted_host(host: str) -> bool:
    """
    Returns True if the host matches the allowlist of trusted RSS providers.
    Allows exact matches or subdomains of trusted providers.
    """
    for provider in TRUSTED_RSS_PROVIDERS:
        if host == provider or host.endswith("." + provider):
            return True
    return False


def check_feed_url(rss_link: str) -> str:
    """
    Validate and sanitize an RSS feed URL.

    Args:
        rss_link: The RSS feed URL string to validate.

    Returns:
        A sanitized, valid RSS feed URL string.

    Raises:
        ValueError: If the URL is not from a trusted provider, uses a disallowed scheme,
                    contains credentials, or targets internal networks.
    """
    if not isinstance(rss_link, str) or not rss_link.strip():
        raise ValueError("RSS link must be a non-empty string")

    parsed = urlparse(rss_link.strip())

    scheme = (parsed.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        raise ValueError("URL scheme is not allowed")

    # Disallow embedded credentials for security
    if parsed.username or parsed.password:
        raise ValueError("Credentials in URL are not allowed")

    host = (parsed.hostname or "").lower()
    if not host:
        raise ValueError("URL must include a valid host")

    # Block internal network targets
    if _is_internal_host(host):
        raise ValueError("Access to internal network addresses is not allowed")

    # Allow only trusted providers (including their subdomains)
    if not _is_trusted_host(host):
        raise ValueError("URL is not from a trusted RSS provider")

    # Rebuild a sanitized URL:
    # - lowercase scheme and host
    # - preserve port, path, and query
    # - drop fragment
    port = f":{parsed.port}" if parsed.port else ""
    path = parsed.path or ""
    query = f"?{parsed.query}" if parsed.query else ""

    sanitized_url = f"{scheme}://{host}{port}{path}{query}"
    return sanitized_url
