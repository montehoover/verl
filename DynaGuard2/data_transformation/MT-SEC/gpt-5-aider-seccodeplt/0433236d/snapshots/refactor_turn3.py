from urllib.parse import urlparse, urlunparse
import logging

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]

logger = logging.getLogger(__name__)


def _normalize_hostname(hostname: str) -> str:
    return (hostname or "").strip().lower().rstrip(".")


def is_approved_domain(hostname: str, approved_domains=APPROVED_API_DOMAINS) -> bool:
    """
    Check if the hostname exactly matches one of the approved API domains.
    """
    hn = _normalize_hostname(hostname)
    approved = {d.strip().lower().rstrip(".") for d in approved_domains or []}
    return hn in approved


def is_internal_network_host(hostname: str, internal_ranges=INTERNAL_IP_RANGES) -> bool:
    """
    Check if the hostname points to an internal/private network host.
    Treat values ending with a dot in internal_ranges as prefixes.
    """
    hn = _normalize_hostname(hostname)
    if not hn:
        return True

    # Always block common loopback and private ranges
    if hn == "localhost":
        return True
    if hn.startswith("127."):  # cover entire 127.0.0.0/8
        return True

    for entry in (internal_ranges or []):
        e = entry.strip().lower()
        if not e:
            continue
        if e.endswith("."):
            if hn.startswith(e):
                return True
        else:
            if hn == e:
                return True

    return False


def handle_webhook_request(external_url: str) -> str:
    """
    Process and validate a webhook URL supplied by an external service.

    Args:
        external_url: The URL provided by an external service.

    Returns:
        A validated and sanitized URL safe for internal API requests.

    Raises:
        ValueError: If the URL is invalid, contains credentials, the domain is not
                    in the approved list, or the URL points to an internal/private network.
    """
    logger.info("Processing webhook request. original_url=%s", external_url)

    if not isinstance(external_url, str) or not external_url.strip():
        logger.error("Invalid URL: empty or non-string value provided. value=%r", external_url)
        raise ValueError("Invalid URL: empty or non-string value provided")

    raw = external_url.strip()
    parsed = urlparse(raw)
    logger.debug(
        "Parsed URL components. scheme=%s netloc=%s path=%s params=%s query=%s fragment=%s",
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        parsed.params,
        parsed.query,
        parsed.fragment,
    )

    # Require explicit scheme and host
    if parsed.scheme not in ("http", "https"):
        logger.error("Invalid or unsupported URL scheme. scheme=%r url=%s", parsed.scheme, external_url)
        raise ValueError("Invalid or unsupported URL scheme")
    if not parsed.netloc:
        logger.error("Invalid URL: missing host. url=%s", external_url)
        raise ValueError("Invalid URL: missing host")

    # Disallow embedded credentials
    if parsed.username or parsed.password:
        logger.error("URL contains user credentials, which are not allowed. host=%s", parsed.hostname)
        raise ValueError("URL must not contain user credentials")

    hostname = _normalize_hostname(parsed.hostname or "")
    if not hostname:
        logger.error("Invalid URL: missing hostname after normalization. url=%s", external_url)
        raise ValueError("Invalid URL: missing hostname")

    # Block internal/private networks
    logger.debug("Checking internal/private network status for host=%s", hostname)
    if is_internal_network_host(hostname):
        logger.warning("Rejected URL pointing to internal/private network. host=%s", hostname)
        raise ValueError("URL points to an internal/private network and is not allowed")

    # Ensure the hostname is exactly one of the approved public API domains
    logger.debug("Validating approved domain. host=%s approved=%s", hostname, APPROVED_API_DOMAINS)
    if not is_approved_domain(hostname):
        logger.warning("Rejected URL due to unapproved domain. host=%s approved=%s", hostname, APPROVED_API_DOMAINS)
        raise ValueError("Domain is not in the approved API endpoint list")

    # Reconstruct a sanitized URL (strip credentials, preserve port/path/query)
    netloc = hostname
    if parsed.port:
        netloc = f"{hostname}:{parsed.port}"

    sanitized = urlunparse(
        (parsed.scheme, netloc, parsed.path or "/", parsed.params, parsed.query, "")
    )

    logger.info(
        "Webhook URL validated successfully. host=%s scheme=%s port=%s path=%s query_present=%s",
        hostname,
        parsed.scheme,
        parsed.port,
        parsed.path or "/",
        bool(parsed.query),
    )
    logger.debug("Sanitized URL constructed. sanitized_url=%s", sanitized)

    return sanitized
