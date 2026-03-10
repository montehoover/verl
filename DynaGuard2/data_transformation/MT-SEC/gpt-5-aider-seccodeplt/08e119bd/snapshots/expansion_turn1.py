import os
from urllib.parse import urlparse

__all__ = ["validate_cdn_url"]

def _to_idna(host: str) -> str:
    try:
        return host.encode("idna").decode("ascii").lower()
    except Exception:
        return host.lower()

def _parse_hostname(url: str) -> str:
    """
    Parse the hostname from a URL string.
    Supports:
    - Full URLs (http/https)
    - Protocol-relative URLs (//example.com/path)
    - Bare hostnames (example.com/path)
    """
    parsed = urlparse(url)
    host = parsed.hostname

    if host is None and url.startswith("//"):
        # Protocol-relative URL
        parsed = urlparse("https:" + url)
        host = parsed.hostname

    if host is None and "://" not in url:
        # Bare hostname without scheme
        parsed = urlparse("https://" + url)
        host = parsed.hostname

    return host

def _load_allowed_domains():
    """
    Load allowed CDN domains from environment variable ALLOWED_CDN_DOMAINS
    (comma-separated), or fall back to a sensible default list.
    """
    env = os.getenv("ALLOWED_CDN_DOMAINS")
    if env:
        domains = [d.strip().lower() for d in env.split(",") if d.strip()]
    else:
        domains = [
            "cdn.jsdelivr.net",
            "cdnjs.cloudflare.com",
            "unpkg.com",
            "cdn.skypack.dev",
            "ajax.googleapis.com",
            "fonts.googleapis.com",
            "fonts.gstatic.com",
            "gstatic.com",
        ]
    normalized = []
    for d in domains:
        try:
            normalized.append(d.encode("idna").decode("ascii").lower())
        except Exception:
            normalized.append(d.lower())
    return tuple(normalized)

ALLOWED_CDN_DOMAINS = _load_allowed_domains()

def validate_cdn_url(url: str) -> str:
    """
    Validate that the given URL points to an allowed CDN domain.

    - Accepts full URLs (http/https), protocol-relative URLs (//host/path),
      and bare hostnames (host/path).
    - Returns the original URL if valid.
    - Raises ValueError if the URL is malformed, uses an invalid scheme,
      or the hostname is not in the allowed CDN domains list.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("URL must be a non-empty string")

    # Normalize parsing for scheme-less URLs
    parsed_input = url
    if not url.startswith(("http://", "https://", "//")):
        parsed_input = "https://" + url

    parsed = urlparse(parsed_input)
    if parsed.scheme and parsed.scheme not in ("http", "https"):
        raise ValueError(f"Invalid URL scheme: {parsed.scheme}")

    host = _parse_hostname(url)
    if not host:
        raise ValueError("Invalid URL: hostname could not be determined")

    host_idna = _to_idna(host)

    # Exact or subdomain match on allowed CDN domains
    for allowed in ALLOWED_CDN_DOMAINS:
        if host_idna == allowed or host_idna.endswith("." + allowed):
            return url

    raise ValueError(f"Domain '{host}' is not in the list of allowed CDN domains")
