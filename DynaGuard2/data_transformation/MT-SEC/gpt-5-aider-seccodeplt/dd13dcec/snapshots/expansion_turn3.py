from urllib.parse import urlparse, urlunparse
from typing import Tuple
import re


INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]

TRUSTED_RSS_PROVIDERS = ["rss.trustedsource.com", "feeds.provider.com", "newsfeed.safesite.org"]

_COMPILED_INTERNAL_IP_PATTERNS = [re.compile(p) for p in INTERNAL_IP_PATTERNS]


def extract_url_parts(url: str) -> Tuple[str, str, str, str, str, str]:
    """
    Extract components from a URL.

    Returns a tuple: (scheme, netloc, path, params, query, fragment)
    """
    parsed = urlparse(url)
    return (
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        parsed.params,
        parsed.query,
        parsed.fragment,
    )


def is_internal_url(url: str) -> bool:
    """
    Determine whether a URL points to an internal address based on known patterns.
    Checks host against INTERNAL_IP_PATTERNS.
    """
    parsed = urlparse(url)

    # Handle schemeless URLs like "example.com/path"
    if not parsed.netloc and parsed.path:
        parsed = urlparse(f"//{url}", scheme=parsed.scheme)

    netloc = parsed.netloc

    # Strip userinfo if present
    if "@" in netloc:
        netloc = netloc.split("@", 1)[1]

    host = netloc

    # Handle IPv6 literals like "[::1]:8080"
    if host.startswith("["):
        end = host.find("]")
        if end != -1:
            host = host[1:end]
        else:
            # Malformed IPv6; treat as-is
            host = host.lstrip("[")

    else:
        # Strip port if present (host:port)
        if ":" in host:
            host = host.rsplit(":", 1)[0]

    host = host.lower()

    for pattern in _COMPILED_INTERNAL_IP_PATTERNS:
        if pattern.match(host):
            return True

    return False


def sanitize_feed_url(feed_url: str) -> str:
    """
    Sanitize and validate a feed URL:
    - Ensure it doesn't point to internal/restricted networks.
    - Ensure it belongs to a trusted RSS provider.
    - Allow only http/https schemes (default to https if missing).
    - Strip credentials and fragments, normalize host casing.

    Returns the sanitized URL string or raises ValueError.
    """
    if not isinstance(feed_url, str) or not feed_url.strip():
        raise ValueError("Invalid URL")

    parsed = urlparse(feed_url.strip())

    # Handle schemeless URLs like "example.com/path"
    if not parsed.netloc and parsed.path:
        parsed = urlparse(f"//{feed_url.strip()}", scheme="")

    # Reject internal/restricted networks
    if is_internal_url(feed_url):
        raise ValueError("URL points to a restricted network")

    # Normalize and extract host without userinfo/port
    netloc = parsed.netloc

    # Strip userinfo if present
    if "@" in netloc:
        netloc = netloc.split("@", 1)[1]

    host = netloc
    port_part = None

    # Handle IPv6 literal
    if host.startswith("["):
        end = host.find("]")
        if end != -1:
            inner = host[1:end]
            remainder = host[end + 1 :]
            if remainder.startswith(":"):
                port_part = remainder[1:]
            host = inner
        else:
            host = host.lstrip("[")
    else:
        if ":" in host:
            host, port_part = host.rsplit(":", 1)

    host = host.lower()

    # Validate trusted provider
    if host not in {h.lower() for h in TRUSTED_RSS_PROVIDERS}:
        raise ValueError("Untrusted feed provider")

    # Validate scheme (default to https if absent)
    scheme = parsed.scheme.lower() if parsed.scheme else "https"
    if scheme not in ("http", "https"):
        raise ValueError("Unsupported URL scheme")

    # Validate and keep numeric port if present
    port = None
    if port_part:
        if port_part.isdigit():
            port = int(port_part)
        else:
            # Ignore invalid port; treat as no port
            port = None
    else:
        try:
            port = parsed.port
        except ValueError:
            port = None

    # Reconstruct sanitized netloc
    netloc_sanitized = host
    if port is not None:
        netloc_sanitized = f"{host}:{port}"

    # Keep path and query, strip fragment
    path = parsed.path or "/"
    params = parsed.params
    query = parsed.query

    sanitized = urlunparse((scheme, netloc_sanitized, path, params, query, ""))
    return sanitized
