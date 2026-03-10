from urllib.parse import urlparse, urlunparse
import re

INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]

TRUSTED_RSS_PROVIDERS = ["rss.trustedsource.com", "feeds.provider.com", "newsfeed.safesite.org"]

def extract_url_parts(url: str):
    parsed = urlparse(url)
    return parsed.scheme, parsed.netloc, parsed.path, parsed.params, parsed.query, parsed.fragment

def is_internal_url(url: str) -> bool:
    parsed = urlparse(url)
    host = getattr(parsed, 'hostname', None)

    # If no scheme was provided, try parsing again with a default scheme
    if not host:
        parsed = urlparse(f"http://{url}")
        host = getattr(parsed, 'hostname', None)

    if not host:
        return False

    host = host.lower()
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, host):
            return True
    return False

def filter_rss_feed(url: str) -> str:
    if not isinstance(url, str) or not url.strip():
        raise ValueError("Invalid RSS feed URL")

    candidate = url.strip()
    parsed = urlparse(candidate)

    # If missing scheme/netloc, try with default http
    if not parsed.netloc:
        parsed = urlparse(f"http://{candidate}")

    host = (parsed.hostname or "").lower()
    scheme = (parsed.scheme or "http").lower()

    if not host:
        raise ValueError("Invalid RSS feed URL")

    if scheme not in ("http", "https"):
        raise ValueError("Untrusted or unsupported URL scheme")

    # Disallow credentials in URL
    if parsed.username or parsed.password:
        raise ValueError("Credentials in URL are not allowed")

    # Block internal network URLs
    if is_internal_url(parsed.geturl()):
        raise ValueError("URL points to an internal network location")

    # Ensure host is a trusted RSS provider
    if host not in TRUSTED_RSS_PROVIDERS:
        raise ValueError("RSS feed provider is not trusted")

    # Sanitize and normalize the URL (lowercase scheme/host, drop userinfo)
    netloc = host
    if parsed.port:
        netloc = f"{host}:{parsed.port}"

    sanitized = urlunparse((
        scheme,
        netloc,
        parsed.path or "",
        parsed.params or "",
        parsed.query or "",
        parsed.fragment or ""
    ))
    return sanitized
