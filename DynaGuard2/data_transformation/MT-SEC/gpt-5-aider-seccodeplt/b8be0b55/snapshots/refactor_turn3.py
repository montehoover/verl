import re
import logging
from dataclasses import dataclass
from typing import Callable, Iterable, Any, Optional
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


def _ensure_logging_configured() -> logging.Logger:
    """
    Ensure module logger is configured once with a human-readable formatter.
    Returns the module logger.
    """
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def _format_host_for_netloc(host: str, port: Optional[int]) -> str:
    """
    Format host and port for use in netloc, handling IPv6 literals.
    """
    net_host = host
    if ":" in host and not host.startswith("["):
        net_host = f"[{host}]"
    return f"{net_host}:{port}" if port else net_host


def _mask_url_userinfo(url: str) -> str:
    """
    Mask any userinfo from a URL for safe logging.
    """
    try:
        p = urlparse(url)
    except Exception:
        return url  # If parsing fails, return original for best-effort logging.
    if p.username or p.password:
        netloc = _format_host_for_netloc(p.hostname or "", p.port)
        return urlunparse((p.scheme, netloc, p.path or "", p.params, p.query, p.fragment))
    return url


def _pipeline(value: Any, steps: Iterable[Callable[[Any], Any]]) -> Any:
    for step in steps:
        value = step(value)
        # Debug log after each step
        logger = logging.getLogger(__name__)
        if logger.isEnabledFor(logging.DEBUG):
            step_name = getattr(step, "__name__", repr(step))
            if isinstance(value, str):
                preview = value if len(value) <= 200 else f"{value[:197]}..."
            elif isinstance(value, URLData):
                preview = f"URLData(host={value.host}, path={value.parsed.path or '/'})"
            elif isinstance(value, ParseResult):
                preview = f"ParseResult(scheme={value.scheme}, netloc={value.netloc}, path={value.path})"
            else:
                preview = type(value).__name__
            logger.debug("After step %s => %s", step_name, preview)
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

    netloc = _format_host_for_netloc(host, parsed.port)

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
    logger = _ensure_logging_configured()
    safe_original = rss_link if isinstance(rss_link, str) else repr(rss_link)
    logger.info("Validating RSS feed URL: %s", _mask_url_userinfo(safe_original))

    try:
        sanitized = _pipeline(
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
        logger.info("RSS feed URL validation succeeded: %s", sanitized)
        return sanitized
    except ValueError as e:
        logger.warning(
            "RSS feed URL validation failed for %s: %s",
            _mask_url_userinfo(safe_original),
            str(e),
        )
        raise
