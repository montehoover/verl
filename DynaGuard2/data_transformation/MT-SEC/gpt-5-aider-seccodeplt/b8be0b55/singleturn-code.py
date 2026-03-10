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


def _is_internal_host(hostname: str) -> bool:
    """
    Returns True if the hostname matches any known internal IP/host patterns.
    """
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            return True
    return False


def _is_trusted_host(hostname: str) -> bool:
    """
    Returns True if the hostname matches a trusted provider or its subdomain.
    """
    hn = hostname.lower()
    for provider in TRUSTED_RSS_PROVIDERS:
        p = provider.lower()
        if hn == p or hn.endswith("." + p):
            return True
    return False


def clean_feed_link(rss_link: str) -> str:
    """
    Validate and sanitize an RSS feed URL to ensure it comes from a trusted provider
    and does not target internal network addresses.

    Args:
        rss_link: The RSS feed URL to sanitize.

    Returns:
        The sanitized feed URL if valid and trusted.

    Raises:
        ValueError: If the URL is invalid, uses a disallowed scheme, includes credentials,
                    matches an internal network address, or is not from a trusted provider.
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
        raise ValueError("URL scheme not allowed. Allowed schemes are: http, https.")

    # Must have a network location
    if not parsed.netloc:
        raise ValueError("URL must include a network location (host).")

    # Disallow embedding credentials
    if parsed.username or parsed.password:
        raise ValueError("Credentials in URL are not allowed.")

    hostname = (parsed.hostname or "").strip().lower()
    if not hostname:
        raise ValueError("URL must include a valid hostname.")

    # Block internal network patterns explicitly
    if _is_internal_host(hostname):
        raise ValueError("Access to internal network addresses is not allowed.")

    # Restrict to trusted providers (allow their subdomains)
    if not _is_trusted_host(hostname):
        raise ValueError("RSS provider is not trusted.")

    # Validate/normalize port if present
    try:
        port = parsed.port  # Accessing .port will validate numeric range if present
    except ValueError:
        raise ValueError("Invalid port in URL.") from None

    # Rebuild a sanitized URL (lowercased scheme/host, no fragment, no userinfo)
    netloc_host = hostname
    netloc = f"{netloc_host}:{port}" if port else netloc_host

    sanitized = urlunparse((
        scheme,
        netloc,
        parsed.path or "",
        "",                  # params (deprecated; clear it)
        parsed.query or "",
        ""                   # fragment removed
    ))

    return sanitized
