from urllib.parse import urlparse, urlunparse
import logging

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]

# Configure module-level logger with clear, readable formatting.
logger = logging.getLogger("callback_validator")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s %(name)s - %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def _ensure_non_empty_string(value: str, name: str) -> str:
    """
    Ensure the provided value is a non-empty string and return its stripped form.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string URL")
    return value.strip()


def _parse_url(url_str: str):
    """
    Parse the URL string into components.
    """
    return urlparse(url_str)


def _validate_scheme(parsed) -> None:
    """
    Validate that the URL scheme is http or https.
    """
    if parsed.scheme not in ("http", "https"):
        raise ValueError("URL scheme must be http or https")


def _reject_userinfo(parsed) -> None:
    """
    Reject URLs that contain embedded user information to prevent credential leakage.
    """
    if parsed.username is not None or parsed.password is not None or ("@" in parsed.netloc):
        raise ValueError("URL must not contain user information")


def _extract_normalized_hostname(parsed) -> str:
    """
    Extract and normalize the hostname to lowercase; ensure it exists.
    """
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise ValueError("URL must include a valid hostname")
    return hostname


def is_internal_or_private_hostname(hostname: str) -> bool:
    """
    Determine if the hostname targets an internal/private address based on INTERNAL_IP_RANGES.
    - Exact match for tokens without trailing dots (e.g., 'localhost', '127.0.0.1').
    - Prefix match for tokens that end with a dot (e.g., '10.', '192.168.').
    """
    for token in INTERNAL_IP_RANGES:
        if token.endswith("."):
            if hostname.startswith(token):
                return True
        else:
            if hostname == token:
                return True
    return False


def is_approved_public_api_domain(hostname: str) -> bool:
    """
    Check if the hostname exactly matches one of the approved public API domains.
    """
    return hostname in APPROVED_API_DOMAINS


def _build_secure_normalized_url(parsed, hostname: str) -> str:
    """
    Build a normalized URL suitable for internal calls:
    - Enforce https scheme
    - Retain port if explicitly provided
    - Preserve path, params, and query
    - Drop fragments
    """
    secure_scheme = "https"
    netloc = hostname
    if parsed.port:
        netloc = f"{hostname}:{parsed.port}"

    return urlunparse(
        (
            secure_scheme,
            netloc,
            parsed.path or "",
            parsed.params or "",
            parsed.query or "",
            "",  # remove fragment
        )
    )


def validate_callback(callback_url: str) -> str:
    """
    Validate and sanitize a callback URL for internal API use.

    - Ensures the URL uses http/https scheme (and enforces https for security).
    - Ensures the hostname is exactly one of the approved API domains.
    - Blocks internal/private network targets based on INTERNAL_IP_RANGES.
    - Rejects URLs with embedded credentials (userinfo).
    - Returns a normalized, secure (https) URL suitable for internal calls.

    Raises:
        ValueError: If the URL is invalid, uses disallowed domains, targets
                    internal/private networks, or contains userinfo.
    """
    logger.info(f"[validate_callback] incoming url={callback_url!r}")
    try:
        cleaned = _ensure_non_empty_string(callback_url, "callback_url")
        parsed = _parse_url(cleaned)
        _validate_scheme(parsed)
        _reject_userinfo(parsed)
        hostname = _extract_normalized_hostname(parsed)

        if is_internal_or_private_hostname(hostname):
            raise ValueError("Access to internal/private network addresses is not allowed")

        if not is_approved_public_api_domain(hostname):
            raise ValueError("URL domain is not in the approved API endpoint list")

        normalized = _build_secure_normalized_url(parsed, hostname)
        logger.info(f"[validate_callback] validation_ok normalized_url={normalized!r} hostname={hostname}")
        return normalized
    except ValueError as e:
        # Best-effort context extraction for logging
        try:
            parsed_for_log = urlparse(callback_url or "")
            hostname_for_log = (parsed_for_log.hostname or "").lower()
        except Exception:
            hostname_for_log = ""
        logger.error(f"[validate_callback] validation_failed url={callback_url!r} hostname={hostname_for_log or '-'} reason={e}")
        raise
