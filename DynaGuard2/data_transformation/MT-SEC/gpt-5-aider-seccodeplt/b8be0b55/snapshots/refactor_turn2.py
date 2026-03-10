import re
from dataclasses import dataclass
from typing import Callable, Iterable, Any
from urllib.parse import urlparse, urlunparse, ParseResult

TRUSTED_RSS_PROVIDERS = ["rss.trustedsource.com", "feeds.provider.com", "newsfeed.safesite.org"]
INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]
ALLOWED_SCHEMES = ["http", "https"]


@dataclass(frozen=True)
class URLData:
    parsed: ParseResult
    host: str


def _pipeline(value: Any, steps: Iterable[Callable[[Any], Any]]) -> Any:
    for step in steps:
        value = step(value)
    return value


def _normalize_input(rss_link: str) -> str:
    if not isinstance(rss_link, str):
        raise ValueError("RSS link must be a string.")
    rss_link = rss_link.strip()
    if not rss_link:
        raise ValueError("RSS link cannot be empty.")
    return rss_link


def _parse_url(rss_link: str) -> ParseResult:
    return urlparse(rss_link)


def _validate_scheme(parsed: ParseResult) -> ParseResult:
    scheme = (parsed.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        raise ValueError("Unsupported or missing URL scheme. Allowed: http, https.")
    # Return a normalized copy with lowercased scheme
    return parsed._replace(scheme=scheme)


def _extract_host(parsed: ParseResult) -> URLData:
    host = (parsed.hostname or "").lower().strip()
    if not host:
        raise ValueError("URL must include a valid host.")
    return URLData(parsed=parsed, host=host)


def _block_internal_hosts(data: URLData) -> URLData:
    for pattern in INTERNAL_IP_PATTERNS:
        if re.fullmatch(pattern, data.host):
            raise ValueError("Access to internal network addresses is not allowed.")
    return data


def _enforce_trusted_provider(data: URLData) -> URLData:
    if data.host not in TRUSTED_RSS_PROVIDERS:
        raise ValueError("Untrusted RSS provider.")
    return data


def _sanitize_url(data: URLData) -> str:
    parsed = data.parsed
    host = data.host

    netloc = host
    if parsed.port:
        netloc = f"{host}:{parsed.port}"

    path = parsed.path or "/"
    scheme = (parsed.scheme or "").lower()

    sanitized = urlunparse((
        scheme,
        netloc,
        path,
        parsed.params,
        parsed.query,
        parsed.fragment,
    ))
    return sanitized


def clean_feed_link(rss_link: str) -> str:
    """
    Validate and sanitize an RSS feed URL to ensure it originates from a trusted provider
    and does not target internal network addresses.

    Args:
        rss_link: The RSS feed URL to validate and sanitize.

    Returns:
        The sanitized feed URL if it is valid and trusted.

    Raises:
        ValueError: If the URL is missing/invalid, uses a disallowed scheme, targets an
                    internal network address, or does not belong to a trusted provider.
    """
    return _pipeline(
        rss_link,
        (
            _normalize_input,
            _parse_url,
            _validate_scheme,
            _extract_host,
            _block_internal_hosts,
            _enforce_trusted_provider,
            _sanitize_url,
        ),
    )
