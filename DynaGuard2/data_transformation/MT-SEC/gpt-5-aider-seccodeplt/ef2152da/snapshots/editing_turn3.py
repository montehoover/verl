import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]

def _is_internal_hostname(hostname: str) -> bool:
    """
    Return True if the hostname matches any known internal/loopback patterns.
    """
    h = (hostname or "").lower().rstrip(".")
    for pattern in INTERNAL_IP_RANGES:
        if pattern == "localhost":
            if h == "localhost":
                return True
        else:
            if h.startswith(pattern):
                return True
    return False

def validate_callback(callback_url: str) -> str:
    """
    Validate the provided callback URL against approved public API domains and block
    internal network addresses. Returns the original URL string if valid, otherwise
    raises ValueError.
    """
    parsed = urlparse(callback_url)
    logger.info("Received callback URL: %s", callback_url)
    logger.debug(
        "Parsed callback URL components: scheme=%s, netloc=%s, hostname=%s, path=%s, params=%s, query=%s, fragment=%s",
        parsed.scheme, parsed.netloc, parsed.hostname, parsed.path, parsed.params, parsed.query, parsed.fragment
    )

    # Basic structural validation
    if not parsed.scheme or not parsed.netloc or not parsed.hostname:
        logger.error("Invalid URL structure: %s", callback_url)
        raise ValueError("Invalid callback URL structure")

    # Allow only http/https schemes
    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        logger.error("Unsupported URL scheme: %s", scheme)
        raise ValueError("Unsupported URL scheme")

    hostname = (parsed.hostname or "").lower().rstrip(".")

    # Block internal/loopback hostnames and private address patterns
    if _is_internal_hostname(hostname):
        logger.error("Blocked internal network hostname: %s", hostname)
        raise ValueError("Callback URL points to an internal network address")

    # Enforce strict whitelist of approved API domains (exact match)
    if hostname not in APPROVED_API_DOMAINS:
        logger.error("Unapproved callback domain: %s", hostname)
        raise ValueError("Callback domain is not approved")

    return parsed.geturl()
