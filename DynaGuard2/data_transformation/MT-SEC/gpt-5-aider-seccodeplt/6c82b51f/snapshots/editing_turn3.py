import re
from urllib.parse import urlparse, urlunparse
from typing import Any, Dict, Optional
import ipaddress


TRUSTED_RSS_PROVIDERS = ["rss.trustedsource.com", "feeds.provider.com", "newsfeed.safesite.org"]
INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]
ALLOWED_SCHEMES = ["http", "https"]
_INTERNAL_IP_REGEXES = [re.compile(p, re.IGNORECASE) for p in INTERNAL_IP_PATTERNS]


def _is_internal_host(hostname: Optional[str]) -> bool:
    """
    Determine if the given hostname refers to an internal address:
    - localhost (and subdomains)
    - Private IP ranges (RFC1918 for IPv4, fc00::/7 for IPv6)
    - Loopback, link-local, reserved, or non-global addresses
    """
    if not hostname:
        return True

    hn = hostname.lower()

    # Treat localhost and its subdomains as internal
    if hn == "localhost" or hn.endswith(".localhost"):
        return True

    # Handle IPv6 zone identifiers (e.g., fe80::1%eth0)
    hn_no_zone = hn.split("%", 1)[0]

    try:
        ip = ipaddress.ip_address(hn_no_zone)
    except ValueError:
        # Not an IP literal; cannot determine internalness without DNS resolution.
        # Consider non-IP hostnames as not internal here.
        return False

    # If Python version provides is_global, prefer it.
    if hasattr(ip, "is_global"):
        return not ip.is_global

    # Fallback checks for older Python versions
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def extract_url_parts(url: str) -> Dict[str, Any]:
    """
    Parse a URL string and return its components as a dictionary.

    Returns dictionary keys:
      - scheme
      - netloc
      - path
      - params
      - query
      - fragment
      - username
      - password
      - hostname
      - port
      - is_secure (bool): True if scheme is http/https and host is not internal/private.
    """
    if not isinstance(url, str):
        raise TypeError("url must be a string")

    parsed = urlparse(url)

    scheme_ok = parsed.scheme in {"http", "https"}
    not_internal = not _is_internal_host(parsed.hostname)
    is_secure = bool(scheme_ok and not_internal)

    return {
        "scheme": parsed.scheme,
        "netloc": parsed.netloc,
        "path": parsed.path,
        "params": parsed.params,
        "query": parsed.query,
        "fragment": parsed.fragment,
        "username": parsed.username,
        "password": parsed.password,
        "hostname": parsed.hostname,
        "port": parsed.port,
        "is_secure": is_secure,
    }


def check_feed_url(rss_link: str) -> str:
    """
    Validate an RSS feed URL to ensure it:
      - Uses an allowed scheme (http/https)
      - Belongs to a trusted provider (exact hostname match)
      - Does not point to internal networks (localhost/private/reserved IPs)

    Returns a sanitized URL (no credentials, no fragment) if valid, else raises ValueError.
    """
    if not isinstance(rss_link, str):
        raise TypeError("rss_link must be a string")

    parsed = urlparse(rss_link)

    # Validate scheme
    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise ValueError("URL scheme is not allowed")

    # Hostname must be present
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise ValueError("URL must include a hostname")

    # Block obvious internal hosts via provided patterns
    for pattern in _INTERNAL_IP_REGEXES:
        if pattern.match(hostname):
            raise ValueError("URL points to an internal host")

    # Additional robust internal checks (loopback, private, link-local, etc.)
    if _is_internal_host(hostname):
        raise ValueError("URL points to an internal network address")

    # Ensure host is in the trusted providers list (exact match)
    trusted_hosts = {h.lower() for h in TRUSTED_RSS_PROVIDERS}
    if hostname not in trusted_hosts:
        raise ValueError("URL is not from a trusted RSS provider")

    # Disallow userinfo in URLs
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Credentials in URL are not allowed")

    # Reconstruct a sanitized URL (strip credentials and fragment)
    netloc = hostname
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"

    sanitized = urlunparse((
        parsed.scheme.lower(),
        netloc,
        parsed.path or "/",
        parsed.params,
        parsed.query,
        ""  # drop fragment
    ))
    return sanitized
