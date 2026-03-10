import re
import logging
from urllib.parse import urlparse, urlunparse, ParseResult

logger = logging.getLogger(__name__)

TRUSTED_RSS_PROVIDERS = ["rss.trustedsource.com", "feeds.provider.com", "newsfeed.safesite.org"]
INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.\d{1,3}\.\d{1,3}$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]
ALLOWED_SCHEMES = ["http", "https"]


def _mask_userinfo_in_url(url: str) -> str:
    # Mask any userinfo present between scheme:// and @
    return re.sub(r"(://)([^/?#@]*@)", r"\1***@", url)


def _normalize_input(url: str) -> str:
    if not isinstance(url, str):
        logger.warning("URL normalization failed: non-string input of type=%s", type(url).__name__)
        raise ValueError("URL must be a string")
    url_str = url.strip()
    if not url_str:
        logger.warning("URL normalization failed: empty string after stripping")
        raise ValueError("URL cannot be empty")
    logger.debug("Input URL normalized: %s", _mask_userinfo_in_url(url_str))
    return url_str


def _parse_url(url_str: str) -> ParseResult:
    parsed = urlparse(url_str)
    logger.debug(
        "URL parsed: scheme=%s, host=%s, port=%s, path_present=%s, query_present=%s, fragment_present=%s",
        (parsed.scheme or "").lower(),
        (parsed.hostname or "").lower() if parsed.hostname else "",
        parsed.port if hasattr(parsed, "port") else None,
        bool(parsed.path),
        bool(parsed.query),
        bool(parsed.fragment),
    )
    return parsed


def _validate_scheme(parsed: ParseResult, allowed_schemes=ALLOWED_SCHEMES) -> str:
    scheme = (parsed.scheme or "").lower()
    if scheme not in allowed_schemes:
        logger.warning("Scheme validation failed: scheme=%s not in %s", scheme, allowed_schemes)
        raise ValueError("URL scheme not allowed")
    logger.debug("Scheme validated: %s", scheme)
    return scheme


def _extract_host(parsed: ParseResult) -> str:
    host = (parsed.hostname or "").lower()
    if not host:
        logger.warning("Host extraction failed: hostname missing in URL")
        raise ValueError("URL must include a hostname")
    logger.debug("Host extracted: %s", host)
    return host


def _ensure_not_internal(host: str, patterns=INTERNAL_IP_PATTERNS) -> str:
    for pattern in patterns:
        if re.match(pattern, host):
            logger.warning("Internal network access blocked: host=%s matched pattern=%s", host, pattern)
            raise ValueError("Access to internal networks is prohibited")
    logger.debug("Host not internal: %s", host)
    return host


def _ensure_trusted_provider(host: str, trusted=TRUSTED_RSS_PROVIDERS) -> str:
    trusted_hosts = {h.lower() for h in trusted}
    if host not in trusted_hosts:
        logger.warning("Untrusted provider rejected: host=%s", host)
        raise ValueError("Untrusted RSS provider")
    logger.debug("Trusted provider verified: %s", host)
    return host


def _sanitize_url(parsed: ParseResult, scheme: str, host: str) -> str:
    try:
        port = parsed.port
    except ValueError:
        logger.warning("Port parsing failed for host=%s in URL", host)
        raise ValueError("Invalid port in URL")

    removed_default_port = False
    if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
        port = None
        removed_default_port = True

    netloc = host if port is None else f"{host}:{port}"

    sanitized_url = urlunparse((
        scheme,
        netloc,
        parsed.path or "",
        parsed.params or "",
        parsed.query or "",
        ""  # strip fragment
    ))

    if removed_default_port:
        logger.debug("Sanitized URL (removed default port): %s", sanitized_url)
    else:
        logger.debug("Sanitized URL: %s", sanitized_url)

    return sanitized_url


def filter_rss_feed(rss_feed_url: str) -> str:
    """
    Validate and sanitize an RSS feed URL to ensure it only targets trusted providers
    and does not access internal networks. Emits logs for each validation step.

    Args:
        rss_feed_url: The URL of the RSS feed being validated.

    Returns:
        A sanitized feed URL limited to trusted providers.

    Raises:
        ValueError: If the URL is invalid, uses a disallowed scheme, targets an internal
                    network, or is not from an approved provider.
    """
    logger.info("Starting RSS URL validation attempt: url=%s", _mask_userinfo_in_url(str(rss_feed_url)))

    try:
        url_str = _normalize_input(rss_feed_url)

        # Pipeline steps with detailed logging inside helpers
        parsed = _parse_url(url_str)
        scheme = _validate_scheme(parsed)
        host = _extract_host(parsed)
        _ensure_not_internal(host)
        _ensure_trusted_provider(host)
        sanitized = _sanitize_url(parsed, scheme, host)

        logger.info("RSS URL validation successful: scheme=%s, host=%s", scheme, host)
        return sanitized
    except ValueError as e:
        logger.error("RSS URL validation failed: %s", str(e))
        raise
