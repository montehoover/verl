import logging
import re
from typing import Iterable, Optional
from urllib.parse import urlparse, urlunparse

TRUSTED_RSS_PROVIDERS = [
    "rss.trustedsource.com",
    "feeds.provider.com",
    "newsfeed.safesite.org",
]
INTERNAL_IP_PATTERNS = [
    r"^localhost$",  # Localhost
    r"^127\.0\.0\.\d+$",  # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$",  # Private 192.168.x.x range
]
ALLOWED_SCHEMES = ["http", "https"]

logger = logging.getLogger(__name__)


def _parse_url(raw_url: str):
    """
    Parse a raw URL string into a urllib.parse.ParseResult.

    Raises:
        ValueError: If the input is not a non-empty string.
    """
    if not isinstance(raw_url, str) or not raw_url.strip():
        raise ValueError("feed_url must be a non-empty string")
    return urlparse(raw_url.strip())


def _validate_scheme(scheme: str, allowed: Iterable[str]) -> str:
    """
    Validate the URL scheme and return a normalized (lowercased) scheme.

    Raises:
        ValueError: If the scheme is not allowed.
    """
    normalized = (scheme or "").lower()
    if normalized not in allowed:
        raise ValueError("URL scheme must be http or https")
    return normalized


def _ensure_netloc(netloc: str) -> None:
    """
    Ensure the URL contains a network location (host).
    """
    if not netloc:
        raise ValueError("URL must include a network location (host)")


def _ensure_no_userinfo(
    username: Optional[str], password: Optional[str]
) -> None:
    """
    Disallow embedded credentials in the URL.
    """
    if username is not None or password is not None:
        raise ValueError("Credentials in URL are not allowed")


def _normalize_hostname(hostname: Optional[str]) -> str:
    """
    Normalize and validate the hostname:
    - Must be present
    - Convert to IDNA for safe comparison
    - Strip trailing dot and lowercase
    - Reject IPv6 literals

    Returns:
        A normalized ASCII hostname.

    Raises:
        ValueError: On invalid or missing hostname.
    """
    if not hostname:
        raise ValueError("URL must include a valid hostname")

    try:
        host_idna = hostname.encode("idna").decode("ascii")
    except Exception as exc:
        raise ValueError("Invalid hostname encoding") from exc

    host_idna = host_idna.rstrip(".").lower()

    # Reject IPv6 literals (trusted list contains DNS names only)
    if ":" in host_idna:
        raise ValueError("IPv6 literal hosts are not allowed")

    return host_idna


def _ensure_not_internal(hostname: str, patterns: Iterable[str]) -> None:
    """
    Ensure the hostname does not match internal/reserved network patterns.

    Raises:
        ValueError: If the hostname matches a restricted pattern.
    """
    for pattern in patterns:
        if re.fullmatch(pattern, hostname):
            raise ValueError("URL targets a restricted internal network address")


def _ensure_trusted(hostname: str, trusted_hosts: Iterable[str]) -> None:
    """
    Ensure the hostname is a trusted provider (exact match or subdomain).

    Raises:
        ValueError: If the host is not trusted.
    """
    trusted = [h.lower() for h in trusted_hosts]
    is_trusted = any(hostname == t or hostname.endswith("." + t) for t in trusted)
    if not is_trusted:
        raise ValueError("URL is not from a trusted RSS provider")


def _validate_port(parsed) -> Optional[int]:
    """
    Validate the port via the parsed URL object. Accessing .port performs
    validation and may raise ValueError.

    Returns:
        The integer port value or None.

    Raises:
        ValueError: If the port is invalid.
    """
    try:
        return parsed.port
    except ValueError as exc:
        raise ValueError("Invalid port in URL") from exc


def _build_netloc(hostname: str, port: Optional[int]) -> str:
    """
    Build the netloc component from hostname and optional port.
    """
    return f"{hostname}:{port}" if port else hostname


def _safe_url_for_logging(parsed) -> str:
    """
    Produce a redacted URL string safe for logs:
    - No credentials
    - No query or fragment
    """
    if not parsed:
        return "<invalid-url>"
    scheme = (parsed.scheme or "").lower()
    hostname = (parsed.hostname or "").rstrip(".").lower()
    try:
        port = parsed.port
    except ValueError:
        port = None
    netloc = f"{hostname}:{port}" if port else hostname
    path = parsed.path or "/"
    return urlunparse((scheme, netloc, path, parsed.params, "", ""))


def sanitize_feed_url(feed_url: str) -> str:
    """
    Validate and sanitize an RSS feed URL to ensure it:
    - Uses an allowed scheme (http or https)
    - Does not target restricted/internal network addresses
    - Belongs to trusted RSS providers (either the exact host or their subdomains)
    - Contains no userinfo credentials

    Args:
        feed_url: The RSS feed URL to be sanitized.

    Returns:
        A sanitized and validated RSS feed URL (scheme and hostname normalized).

    Raises:
        ValueError: If the URL is invalid, uses a disallowed scheme, includes
                    credentials, targets restricted network locations, or is not
                    from a trusted provider.
    """
    try:
        parsed = _parse_url(feed_url)
        redacted = _safe_url_for_logging(parsed)
        logger.info("Validating RSS feed URL: %s", redacted)

        scheme = _validate_scheme(parsed.scheme, ALLOWED_SCHEMES)
        _ensure_netloc(parsed.netloc)
        _ensure_no_userinfo(parsed.username, parsed.password)

        host_idna = _normalize_hostname(parsed.hostname)
        _ensure_not_internal(host_idna, INTERNAL_IP_PATTERNS)
        _ensure_trusted(host_idna, TRUSTED_RSS_PROVIDERS)

        port = _validate_port(parsed)
        netloc = _build_netloc(host_idna, port)

        sanitized = urlunparse(
            (
                scheme,
                netloc,
                parsed.path or "/",
                parsed.params,
                parsed.query,
                "",  # drop fragment
            )
        )

        logger.info(
            "RSS feed URL validated successfully: %s",
            _safe_url_for_logging(urlparse(sanitized)),
        )
        return sanitized
    except ValueError as exc:
        # Attempt a safe representation for logging
        try:
            parsed_for_log = (
                urlparse(feed_url.strip()) if isinstance(feed_url, str) else None
            )
        except Exception:
            parsed_for_log = None
        redacted = _safe_url_for_logging(parsed_for_log)
        logger.warning(
            "RSS feed URL validation failed for %s: %s", redacted, str(exc)
        )
        raise
