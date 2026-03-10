import re
from urllib.parse import urlparse, urlunparse
from typing import Dict


def extract_url_parts(url: str) -> Dict[str, str]:
    """
    Extract components from a URL.

    Returns a dictionary with keys: scheme, netloc, path, params, query, fragment.
    """
    parsed = urlparse(url)
    return {
        "scheme": parsed.scheme,
        "netloc": parsed.netloc,
        "path": parsed.path,
        "params": parsed.params,
        "query": parsed.query,
        "fragment": parsed.fragment,
    }


INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]

_INTERNAL_IP_REGEX = [re.compile(p, re.IGNORECASE) for p in INTERNAL_IP_PATTERNS]


def is_unsafe_url(url: str) -> bool:
    """
    Determine if a URL points to an internal/unsafe address based on hostname patterns.
    Matches against INTERNAL_IP_PATTERNS.
    """
    parsed = urlparse(url)
    host = parsed.hostname

    # Handle URLs without a scheme, e.g., "localhost:8000" or "192.168.1.10/path"
    if host is None and "://" not in url:
        host = urlparse("//" + url).hostname

    if not host:
        return False

    host = host.lower()

    return any(regex.match(host) for regex in _INTERNAL_IP_REGEX)


TRUSTED_RSS_PROVIDERS = ["rss.trustedsource.com", "feeds.provider.com", "newsfeed.safesite.org"]


def clean_feed_link(feed_url: str) -> str:
    """
    Validate and sanitize an RSS feed URL.

    - Ensures the URL does not point to internal networks.
    - Only allows http/https schemes.
    - Verifies the hostname is in TRUSTED_RSS_PROVIDERS.
    - Returns a sanitized URL (lowercased scheme/host, removes credentials/fragment).
    """
    if not feed_url or not feed_url.strip():
        raise ValueError("Feed URL is empty")

    url = feed_url.strip()

    # Parse URL; if scheme is missing, assume http to allow hostname extraction and validation.
    parsed = urlparse(url)
    if not parsed.scheme and "://" not in url:
        url = "http://" + url
        parsed = urlparse(url)

    # Block internal/unsafe hosts early
    if is_unsafe_url(url):
        raise ValueError("Feed URL points to an internal or unsafe address")

    # Only allow http/https
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("Unsupported feed URL scheme")

    host = parsed.hostname
    if not host:
        raise ValueError("Invalid feed URL")

    host_l = host.lower()

    # Verify trusted provider
    if host_l not in TRUSTED_RSS_PROVIDERS:
        raise ValueError("Untrusted RSS provider")

    # Sanitize: drop credentials and fragment, normalize host/scheme, keep path/params/query.
    scheme = parsed.scheme.lower()
    port = parsed.port
    default_ports = {"http": 80, "https": 443}
    if port and port != default_ports.get(scheme):
        netloc = f"{host_l}:{port}"
    else:
        netloc = host_l

    path = parsed.path or "/"
    params = parsed.params
    query = parsed.query
    fragment = ""  # drop fragment

    sanitized = urlunparse((scheme, netloc, path, params, query, fragment))
    return sanitized
