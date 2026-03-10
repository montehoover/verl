from urllib.parse import urlparse, urlunparse
import logging

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]

logger = logging.getLogger(__name__)


def is_internal_host(host: str, internal_ranges=INTERNAL_IP_RANGES) -> bool:
    """
    Determine if a hostname points to an internal or loopback location.
    Returns True if the host is internal, False otherwise.
    """
    host_lc = (host or "").lower()
    for rng in internal_ranges:
        rng_lc = rng.lower()
        if rng_lc.endswith("."):
            if host_lc.startswith(rng_lc):
                return True
        else:
            if host_lc == rng_lc:
                return True
    return False


def is_approved_domain(host: str, approved_domains=APPROVED_API_DOMAINS) -> bool:
    """
    Check if the hostname exactly matches one of the approved API domains.
    """
    approved = {d.lower() for d in approved_domains}
    return (host or "").lower() in approved


def _sanitize_url_for_log(url: str) -> str:
    """
    Sanitize a URL for logging by removing user credentials and fragments.
    """
    try:
        parsed = urlparse(url)
        # If it's not a URL, just return the stripped string
        if not parsed.scheme and not parsed.netloc:
            return url.strip()
        netloc = parsed.hostname or ""
        if parsed.port:
            netloc = f"{netloc}:{parsed.port}"
        return urlunparse(
            (
                parsed.scheme,
                netloc,
                parsed.path or "",
                parsed.params,
                parsed.query,
                "",  # drop fragment
            )
        )
    except Exception:
        return str(url)


def process_hook(hook_url: str) -> str:
    """
    Validate a webhook URL and return a safely validated URL that references an approved API endpoint.

    Raises:
        ValueError: If the URL is invalid, uses a non-http(s) scheme, points to an internal location,
                    or the domain is not in the approved list.
    """
    if not isinstance(hook_url, str):
        logger.warning("Rejected hook_url: expected str, got %s", type(hook_url).__name__)
        raise ValueError("hook_url must be a string")

    candidate = hook_url.strip()
    if not candidate:
        logger.warning("Rejected hook_url: empty string")
        raise ValueError("hook_url must not be empty")

    log_url = _sanitize_url_for_log(candidate)
    logger.info("Processing hook URL: %s", log_url)

    parsed = urlparse(candidate)

    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        logger.warning("Rejected URL (invalid scheme='%s'): %s", scheme or "<none>", log_url)
        raise ValueError("URL scheme must be http or https")
    logger.debug("Scheme validated: %s", scheme)

    host = (parsed.hostname or "").lower()
    if not host:
        logger.warning("Rejected URL (missing hostname): %s", log_url)
        raise ValueError("URL must include a valid hostname")
    logger.debug("Hostname extracted: %s", host)

    logger.debug("Validating that host is not internal: %s", host)
    if is_internal_host(host):
        logger.warning("Rejected URL (targets internal network location): %s", log_url)
        raise ValueError("URL targets an internal network location")
    logger.debug("Internal network check passed for host: %s", host)

    logger.debug("Validating that host is an approved API domain: %s", host)
    if not is_approved_domain(host):
        logger.warning("Rejected URL (unapproved domain): %s", log_url)
        raise ValueError("URL domain is not an approved API endpoint")
    logger.debug("Approved domain check passed for host: %s", host)

    sanitized = urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path or "",
            parsed.params,
            parsed.query,
            "",  # no fragment
        )
    )
    logger.info("URL validated and sanitized: %s", _sanitize_url_for_log(sanitized))
    return sanitized
