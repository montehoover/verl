from urllib.parse import urlparse
from typing import Tuple
import re

INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]

TRUSTED_RSS_PROVIDERS = ["rss.trustedsource.com", "feeds.provider.com", "newsfeed.safesite.org"]

def extract_url_parts(url: str) -> Tuple[str, str, str, str, str, str]:
    """
    Extract and return the scheme, netloc, path, params, query, and fragment from a URL.

    :param url: The URL string to parse.
    :return: A tuple (scheme, netloc, path, params, query, fragment).
    """
    parsed = urlparse(url)
    return parsed.scheme, parsed.netloc, parsed.path, parsed.params, parsed.query, parsed.fragment

def is_internal_network_url(url: str) -> bool:
    """
    Determine whether the given URL points to an internal network host based on hostname patterns.

    :param url: The URL string to check.
    :return: True if the URL hostname matches any internal IP/host patterns, False otherwise.
    """
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().strip("[]")

    if not host:
        return False

    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, host):
            return True
    return False

def validate_rss_link(url: str) -> str:
    """
    Validate that the given RSS feed URL:
      - has an http(s) scheme,
      - points to a trusted provider,
      - does not target internal networks,
    and return a sanitized URL (without credentials).

    :param url: RSS feed URL to validate.
    :return: Sanitized URL string.
    :raises ValueError: If the URL is invalid, untrusted, or internal.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("RSS URL must be a non-empty string.")
    url = url.strip()

    parsed = urlparse(url)

    if not parsed.scheme or not parsed.netloc:
        raise ValueError("Invalid RSS URL format.")

    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise ValueError("RSS URL must use http or https scheme.")

    # Reject internal network targets
    if is_internal_network_url(url):
        raise ValueError("RSS URL points to a restricted internal network host.")

    host = (parsed.hostname or "").lower().strip("[]")
    if not host:
        raise ValueError("RSS URL must contain a valid hostname.")

    trusted_hosts = {h.lower() for h in TRUSTED_RSS_PROVIDERS}
    if host not in trusted_hosts:
        raise ValueError("RSS URL is not from a trusted provider.")

    # Sanitize: remove credentials; keep optional port, path, and query; drop fragment.
    netloc = host
    if parsed.port:
        netloc = f"{host}:{parsed.port}"
    path = parsed.path or "/"
    query = f"?{parsed.query}" if parsed.query else ""

    sanitized = f"{scheme}://{netloc}{path}{query}"
    return sanitized
