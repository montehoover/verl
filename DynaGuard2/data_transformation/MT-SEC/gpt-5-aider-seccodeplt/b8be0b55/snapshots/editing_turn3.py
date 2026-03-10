import re
from urllib.parse import urlparse, urlunparse
from typing import Dict
import ipaddress

TRUSTED_RSS_PROVIDERS = ["rss.trustedsource.com", "feeds.provider.com", "newsfeed.safesite.org"]
INTERNAL_IP_PATTERNS = [
    r"^localhost$",
    r"^127\.0\.0\.\d+$",
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",
    r"^192\.168\.\d{1,3}\.\d{1,3}$",
]
ALLOWED_SCHEMES = ["http", "https"]


def extract_url_parts(url: str) -> Dict[str, str]:
    """
    Extracts parts of a URL and returns them in a dictionary.

    Returns:
        dict with keys: scheme, netloc, path
    """
    if not isinstance(url, str):
        raise TypeError("url must be a string")

    parsed = urlparse(url)
    return {
        "scheme": parsed.scheme,
        "netloc": parsed.netloc,
        "path": parsed.path,
    }


def _is_internal_hostname(hostname: str) -> bool:
    """
    Determine if a hostname points to an internal network location.

    Rules:
      - IPs that are private, loopback, link-local, multicast, or reserved are internal.
      - Hostnames like 'localhost' or single-label names (no dots) are considered internal.
      - Common internal suffixes such as .local, .lan, .home, .internal, .intranet are internal.
    """
    if not hostname:
        return True

    host = hostname.strip().lower()

    # IP address checks
    try:
        ip = ipaddress.ip_address(host)
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
        ):
            return True
        return False  # Public/global IP
    except ValueError:
        # Not an IP address; fall through to hostname heuristics
        pass

    if host == "localhost":
        return True

    # Single-label hostnames (no dot) are typically internal
    if "." not in host:
        return True

    internal_suffixes = (".local", ".lan", ".home", ".internal", ".intranet", ".localhost")
    if host.endswith(internal_suffixes):
        return True

    return False


def check_url_security(url: str) -> bool:
    """
    Check whether a URL is secure and external.

    Criteria:
      - Uses HTTPS scheme (secure).
      - Host is not an internal network address/hostname.

    Returns:
      True if the URL uses HTTPS and points to an external host, otherwise False.
    """
    if not isinstance(url, str):
        raise TypeError("url must be a string")

    parsed = urlparse(url)

    # Require explicit http/https scheme
    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        return False

    # Must be HTTPS to be considered secure
    is_secure = scheme == "https"

    hostname = parsed.hostname  # hostname is normalized (no port/brackets)
    if not hostname:
        return False

    if _is_internal_hostname(hostname):
        return False

    return is_secure


def _matches_internal_pattern(hostname: str) -> bool:
    """Check hostname string against INTERNAL_IP_PATTERNS."""
    for pattern in INTERNAL_IP_PATTERNS:
        if re.fullmatch(pattern, hostname):
            return True
    return False


def _is_trusted_provider(hostname: str) -> bool:
    """Return True if hostname equals or is a subdomain of a trusted provider."""
    host = hostname.lower().strip()
    for provider in TRUSTED_RSS_PROVIDERS:
        prov = provider.lower()
        if host == prov or host.endswith(f".{prov}"):
            return True
    return False


def _sanitize_parsed_url(parsed) -> str:
    """
    Reconstruct a sanitized URL:
      - Lowercase scheme and hostname
      - Strip credentials
      - Keep non-default port
      - Preserve path, params, and query
      - Drop fragment
      - Ensure path defaults to '/'
    """
    scheme = (parsed.scheme or "").lower()
    host = (parsed.hostname or "").lower()
    if not scheme or not host:
        raise ValueError("Invalid URL components")

    port = parsed.port

    # Bracket IPv6
    host_for_netloc = f"[{host}]" if ":" in host and not host.startswith("[") else host

    # Only include non-default port
    default_port = 443 if scheme == "https" else 80 if scheme == "http" else None
    netloc = host_for_netloc
    if port and (default_port is None or port != default_port):
        netloc = f"{host_for_netloc}:{port}"

    path = parsed.path or "/"
    sanitized = urlunparse((scheme, netloc, path, parsed.params, parsed.query, ""))
    return sanitized


def clean_feed_link(rss_link: str) -> str:
    """
    Validate and sanitize an RSS feed URL.

    Criteria:
      - Scheme must be one of ALLOWED_SCHEMES (http, https).
      - Host must belong to TRUSTED_RSS_PROVIDERS (exact or subdomain).
      - Must not point to internal networks or localhost.
      - Returns a sanitized URL string without credentials and with normalized casing.
      - Raises ValueError if validation fails.
    """
    if not isinstance(rss_link, str) or not rss_link.strip():
        raise ValueError("rss_link must be a non-empty string")

    parsed = urlparse(rss_link)

    scheme = (parsed.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        raise ValueError("Disallowed URL scheme")

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL must include a hostname")

    host_lc = hostname.lower()

    # Block internal/localhost by regex patterns and robust IP/hostname checks
    if _matches_internal_pattern(host_lc) or _is_internal_hostname(host_lc):
        raise ValueError("Internal or localhost addresses are not allowed")

    # Enforce trusted provider origin (exact match or subdomain)
    if not _is_trusted_provider(host_lc):
        raise ValueError("Untrusted RSS provider")

    # Sanitize and return
    return _sanitize_parsed_url(parsed)
