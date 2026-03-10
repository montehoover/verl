import re
from urllib.parse import urlparse, urlunparse, ParseResult

TRUSTED_RSS_PROVIDERS = ["rss.trustedsource.com", "feeds.provider.com", "newsfeed.safesite.org"]
INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.\d{1,3}\.\d{1,3}$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]
ALLOWED_SCHEMES = ["http", "https"]


def _normalize_input(url: str) -> str:
    if not isinstance(url, str):
        raise ValueError("URL must be a string")
    url_str = url.strip()
    if not url_str:
        raise ValueError("URL cannot be empty")
    return url_str


def _parse_url(url_str: str) -> ParseResult:
    return urlparse(url_str)


def _validate_scheme(parsed: ParseResult, allowed_schemes=ALLOWED_SCHEMES) -> str:
    scheme = (parsed.scheme or "").lower()
    if scheme not in allowed_schemes:
        raise ValueError("URL scheme not allowed")
    return scheme


def _extract_host(parsed: ParseResult) -> str:
    host = (parsed.hostname or "").lower()
    if not host:
        raise ValueError("URL must include a hostname")
    return host


def _ensure_not_internal(host: str, patterns=INTERNAL_IP_PATTERNS) -> str:
    for pattern in patterns:
        if re.match(pattern, host):
            raise ValueError("Access to internal networks is prohibited")
    return host


def _ensure_trusted_provider(host: str, trusted=TRUSTED_RSS_PROVIDERS) -> str:
    trusted_hosts = {h.lower() for h in trusted}
    if host not in trusted_hosts:
        raise ValueError("Untrusted RSS provider")
    return host


def _sanitize_url(parsed: ParseResult, scheme: str, host: str) -> str:
    try:
        port = parsed.port
    except ValueError:
        raise ValueError("Invalid port in URL")

    if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
        port = None

    netloc = host if port is None else f"{host}:{port}"

    sanitized_url = urlunparse((
        scheme,
        netloc,
        parsed.path or "",
        parsed.params or "",
        parsed.query or "",
        ""  # strip fragment
    ))
    return sanitized_url


def filter_rss_feed(rss_feed_url: str) -> str:
    """
    Validate and sanitize an RSS feed URL to ensure it only targets trusted providers
    and does not access internal networks.

    Args:
        rss_feed_url: The URL of the RSS feed being validated.

    Returns:
        A sanitized feed URL limited to trusted providers.

    Raises:
        ValueError: If the URL is invalid, uses a disallowed scheme, targets an internal
                    network, or is not from an approved provider.
    """
    url_str = _normalize_input(rss_feed_url)

    # Pipeline steps
    parsed = _parse_url(url_str)
    scheme = _validate_scheme(parsed)
    host = _extract_host(parsed)
    _ensure_not_internal(host)
    _ensure_trusted_provider(host)
    return _sanitize_url(parsed, scheme, host)
