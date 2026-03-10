from typing import Any, Dict
from urllib.parse import urlparse, urlunparse
import ipaddress
import re


TRUSTED_RSS_PROVIDERS = ["rss.trustedsource.com", "feeds.provider.com", "newsfeed.safesite.org"]
INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]
ALLOWED_SCHEMES = ["http", "https"]


def _is_public_http_url(parts) -> bool:
    # Scheme must be http or https
    if parts.scheme not in ("http", "https"):
        return False

    # Must have a hostname
    host = parts.hostname
    if not host:
        return False

    # Reject obvious local hostnames
    lowered = host.lower()
    if lowered in {"localhost", "localhost.localdomain"}:
        return False

    # Handle potential IPv6 zone identifiers (e.g., fe80::1%eth0)
    host_for_ip_check = lowered.split("%", 1)[0]

    # If it's an IP literal, ensure it's not private/loopback/link-local/etc.
    try:
        ip = ipaddress.ip_address(host_for_ip_check)
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return False
    except ValueError:
        # Not an IP literal; allow domain names other than localhost
        pass

    return True


def extract_url_parts(url: str) -> Dict[str, Any]:
    """
    Parse a URL string and return its components in a dictionary,
    including a boolean 'is_valid' indicating whether the URL:
      - uses http or https
      - does not point to localhost or private/internal IP ranges
    """
    if not isinstance(url, str):
        raise TypeError("url must be a string")

    parts = urlparse(url)

    result: Dict[str, Any] = {
        "scheme": parts.scheme or None,
        "username": parts.username,
        "password": parts.password,
        "hostname": parts.hostname,
        "port": parts.port,
        "path": parts.path,
        "params": parts.params,
        "query": parts.query,
        "fragment": parts.fragment,
    }

    result["is_valid"] = _is_public_http_url(parts)
    return result


def _normalize_hostname(hostname: str) -> str:
    """
    Normalize a hostname for comparison:
    - lowercase
    - strip trailing dot
    """
    return hostname.rstrip(".").lower()


def _is_internal_hostname(hostname: str) -> bool:
    """
    Determine if the given hostname refers to an internal or special-use address.
    Checks:
      - IP literal (IPv4/IPv6): private/loopback/link-local/unspecified/reserved/multicast
      - Regex patterns for localhost and common private IPv4 ranges
    """
    if not hostname:
        return True

    hn = _normalize_hostname(hostname)
    # Handle potential IPv6 zone identifiers (e.g., fe80::1%eth0)
    hn_no_zone = hn.split("%", 1)[0]

    try:
        ip = ipaddress.ip_address(hn_no_zone)
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return True
    except ValueError:
        # Not an IP literal, fall through to regex checks
        pass

    if hn in {"localhost", "localhost.localdomain"}:
        return True

    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hn):
            return True

    return False


def _is_trusted_provider(hostname: str) -> bool:
    """
    Return True if the hostname matches an approved provider exactly or as a subdomain.
    """
    hn = _normalize_hostname(hostname)
    for provider in TRUSTED_RSS_PROVIDERS:
        provider_norm = _normalize_hostname(provider)
        if hn == provider_norm or hn.endswith("." + provider_norm):
            return True
    return False


def filter_rss_feed(rss_feed_url: str) -> str:
    """
    Validate and sanitize an RSS feed URL.
    - Must use an allowed scheme (http/https)
    - Must not point to internal networks (localhost, private IP ranges, etc.)
    - Must belong to a trusted provider (exact match or subdomain)
    Returns a sanitized URL (lowercased scheme/host, no credentials, no fragment).
    Raises ValueError if validation fails.
    """
    if not isinstance(rss_feed_url, str):
        raise TypeError("rss_feed_url must be a string")

    url = rss_feed_url.strip()
    parts = urlparse(url)

    if not parts.scheme:
        raise ValueError("URL must include a scheme")
    scheme = parts.scheme.lower()
    if scheme not in ALLOWED_SCHEMES:
        raise ValueError("URL scheme is not allowed")

    hostname = parts.hostname  # already lowercased by urllib.parse
    if not hostname:
        raise ValueError("URL must include a hostname")

    # Reject internal addresses
    if _is_internal_hostname(hostname):
        raise ValueError("URL points to an internal or special-use address")

    # Ensure the host is an approved provider
    if not _is_trusted_provider(hostname):
        raise ValueError("URL host is not an approved RSS provider")

    # Sanitize: drop credentials and fragment, normalize host casing, omit default ports
    host_norm = _normalize_hostname(hostname)

    # Determine if IPv6 and wrap in brackets for netloc if necessary
    def _format_host_for_netloc(h: str) -> str:
        try:
            ip = ipaddress.ip_address(h.split("%", 1)[0])
            if ip.version == 6:
                return f"[{h}]"
        except ValueError:
            pass
        return h

    port = parts.port
    default_port = 80 if scheme == "http" else 443
    netloc_host = _format_host_for_netloc(host_norm)
    if port and port != default_port:
        netloc = f"{netloc_host}:{port}"
    else:
        netloc = netloc_host

    path = parts.path or "/"
    query = parts.query or ""

    sanitized = urlunparse((scheme, netloc, path, "", query, ""))
    return sanitized
