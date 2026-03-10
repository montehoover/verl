import logging
from urllib.parse import urlparse, urlunparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]

# Module logger (no-op by default; applications can configure handlers/levels)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def is_internal_or_private_host(host: str, internal_ranges: list[str]) -> bool:
    """
    Determine whether a hostname refers to an internal or private network target.

    This function performs simple string comparisons against known internal
    hostnames and CIDR-like private IP prefixes. It treats an empty host
    as internal/private to err on the side of safety.

    Args:
        host: Hostname to evaluate. Case-insensitive; may be a DNS name or IP.
        internal_ranges: A list of exact hostnames or string prefixes that
            identify internal/private addresses (e.g., "localhost", "10.", "192.168.").

    Returns:
        True if the host should be considered internal/private; otherwise False.

    Examples:
        >>> is_internal_or_private_host("localhost", INTERNAL_IP_RANGES)
        True
        >>> is_internal_or_private_host("10.2.3.4", INTERNAL_IP_RANGES)
        True
        >>> is_internal_or_private_host("api.service.com", INTERNAL_IP_RANGES)
        False
    """
    if not host:
        return True  # Treat empty host as invalid/internal
    host_lc = host.lower()
    for prefix in internal_ranges:
        if host_lc == prefix or host_lc.startswith(prefix):
            return True
    return False


def is_approved_domain(host: str, approved_domains: list[str]) -> bool:
    """
    Check if a hostname exactly matches an approved API domain.

    Args:
        host: Hostname to validate (case-insensitive).
        approved_domains: The allowlist of approved public API domains.

    Returns:
        True if the host is an exact match for one of the approved domains;
        otherwise False.

    Examples:
        >>> is_approved_domain("api.service.com", APPROVED_API_DOMAINS)
        True
        >>> is_approved_domain("sub.api.service.com", APPROVED_API_DOMAINS)
        False
        >>> is_approved_domain("", APPROVED_API_DOMAINS)
        False
    """
    if not host:
        return False
    host_lc = host.lower()
    return any(host_lc == d.lower() for d in approved_domains)


def validate_webhook(webhook_link: str) -> str:
    """
    Validate an external webhook URL and return a secure, internal-allowed URL.

    This function ensures:
      - The input is a non-empty string.
      - The URL parses correctly and has an HTTP/HTTPS scheme.
      - The host is not an internal/private address.
      - The host exactly matches an approved public API domain.
      - The returned URL uses HTTPS and excludes userinfo.

    It also emits structured log messages at DEBUG level for each validation
    step, and logs an ERROR message with context before raising any ValueError
    to aid in tracing validation failures gracefully.

    Args:
        webhook_link: External webhook URL string.

    Returns:
        A validated HTTPS URL string suitable for internal API interactions.

    Raises:
        ValueError: If the URL is invalid, uses an unsupported scheme,
                    targets internal/private network addresses, or the domain
                    is not in the approved API endpoint list.

    Examples:
        >>> validate_webhook("http://api.service.com/hooks?id=123")
        'https://api.service.com/hooks?id=123'
    """
    logger.debug("validate_webhook:start len=%s", (len(webhook_link) if isinstance(webhook_link, str) else "n/a"))

    if not isinstance(webhook_link, str) or not webhook_link.strip():
        logger.error("validate_webhook:error reason=invalid_input url_type=%s", type(webhook_link).__name__)
        raise ValueError("Invalid URL input.")

    raw = webhook_link.strip()
    parsed = urlparse(raw)
    logger.debug(
        "validate_webhook:parsed scheme=%s host=%s port=%s path=%s query_present=%s",
        parsed.scheme or "",
        parsed.hostname or "",
        parsed.port or "",
        parsed.path or "",
        bool(parsed.query),
    )

    # Must have scheme and netloc
    if not parsed.scheme or not parsed.netloc:
        logger.error(
            "validate_webhook:error reason=invalid_format scheme=%s netloc=%s",
            parsed.scheme or "",
            parsed.netloc or "",
        )
        raise ValueError("Invalid URL format.")

    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        logger.error("validate_webhook:error reason=unsupported_scheme scheme=%s", scheme)
        raise ValueError("Unsupported URL scheme.")

    host = (parsed.hostname or "").lower()
    if not host:
        logger.error("validate_webhook:error reason=missing_host")
        raise ValueError("Invalid URL host.")

    # Block internal/private network addresses by prefix check
    if is_internal_or_private_host(host, INTERNAL_IP_RANGES):
        logger.error("validate_webhook:error reason=internal_or_private host=%s", host)
        raise ValueError("URL points to a disallowed internal/private address.")

    # Enforce allowlist of API domains (exact match)
    if not is_approved_domain(host, APPROVED_API_DOMAINS):
        logger.error("validate_webhook:error reason=unapproved_domain host=%s", host)
        raise ValueError("URL domain is not approved.")

    # Force HTTPS for secure internal interactions
    secure_scheme = "https"

    # Reconstruct sanitized netloc without userinfo
    netloc = host
    if parsed.port:
        netloc = f"{host}:{parsed.port}"

    path = parsed.path or "/"
    query = parsed.query or ""

    secure_url = urlunparse((secure_scheme, netloc, path, "", query, ""))

    logger.debug(
        "validate_webhook:success host=%s approved_domain=True internal_check=False secure_url=%s",
        host,
        secure_url,
    )
    return secure_url
