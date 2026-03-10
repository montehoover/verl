import logging
from urllib.parse import urlparse
from typing import Optional

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]


logger = logging.getLogger(__name__)


def _sanitize_url_for_logging(url: str) -> str:
    """
    Return a sanitized representation of a URL for logging purposes.

    This helper removes any user credentials from the netloc while preserving
    other components. If the URL cannot be parsed, a placeholder is returned.

    Args:
        url: Raw URL string that may contain credentials.

    Returns:
        A sanitized URL string safe for logs (no username/password).
    """
    try:
        p = urlparse(url)
    except Exception:
        return "<unparseable URL>"

    hostname = p.hostname or ""
    # Wrap IPv6 literals in brackets for clarity
    if ":" in hostname and not hostname.startswith("["):
        hostname_display = f"[{hostname}]"
    else:
        hostname_display = hostname

    # Port property can raise ValueError if invalid; guard it.
    try:
        port_part = f":{p.port}" if p.port else ""
    except ValueError:
        port_part = ""

    scheme_part = f"{p.scheme}://" if p.scheme else ""
    path_part = p.path or ""
    query_part = f"?{p.query}" if p.query else ""
    fragment_part = f"#{p.fragment}" if p.fragment else ""

    return f"{scheme_part}{hostname_display}{port_part}{path_part}{query_part}{fragment_part}"


def validate_webhook_url(webhook_url: str) -> str:
    """
    Validate a webhook URL against security and policy constraints.

    The validation enforces that the URL:
      - Is a non-empty string.
      - Is a well-formed HTTP(S) URL.
      - Contains no embedded credentials.
      - Does not resolve to or target internal/loopback ranges indicated by configured prefixes.
      - Belongs to an approved API domain (or its subdomain).

    Logging:
      - This function emits DEBUG-level logs detailing each validation step.
      - Sensitive data (credentials) are never logged; URLs are sanitized before logging.

    Args:
        webhook_url: The URL provided by an external service.

    Returns:
        The original URL string if validation succeeds.

    Raises:
        ValueError: If the URL is malformed, uses an unsupported scheme, includes credentials,
                    targets an internal address, or is not within the approved domains.
    """
    sanitized = _sanitize_url_for_logging(webhook_url)
    logger.debug("Starting validation for webhook URL: %s", sanitized)

    if not isinstance(webhook_url, str) or not webhook_url.strip():
        logger.debug("Validation failure: URL is not a non-empty string.")
        raise ValueError("A non-empty webhook URL string is required.")

    parsed = urlparse(webhook_url)
    logger.debug("Parsed URL components: scheme=%s, netloc=%s, path=%s", parsed.scheme, parsed.netloc, parsed.path)

    # Basic structure and scheme checks
    if not parsed.scheme or not parsed.netloc:
        logger.debug("Validation failure: Missing scheme or host. scheme=%s, netloc=%s", parsed.scheme, parsed.netloc)
        raise ValueError("Webhook URL must include scheme and host.")

    if parsed.scheme not in ("http", "https"):
        logger.debug("Validation failure: Unsupported scheme: %s", parsed.scheme)
        raise ValueError("Only http and https schemes are allowed.")

    # Disallow credentials in URL
    if parsed.username or parsed.password:
        logger.debug("Validation failure: Credentials present in URL netloc.")
        raise ValueError("Credentials in URLs are not allowed.")

    # Hostname validation
    hostname = (parsed.hostname or "").lower()
    logger.debug("Hostname extracted for validation: %s", hostname)

    if not hostname:
        logger.debug("Validation failure: Hostname is empty or invalid.")
        raise ValueError("Webhook URL must include a valid hostname.")

    # Block internal/loopback ranges based on provided prefixes
    logger.debug("Evaluating hostname against internal ranges: %s", INTERNAL_IP_RANGES)
    for prefix in INTERNAL_IP_RANGES:
        if hostname == prefix or hostname.startswith(prefix):
            logger.debug("Validation failure: Hostname matches internal prefix: %s", prefix)
            raise ValueError("Internal addresses are not allowed.")

    # Allow exact approved domains and their subdomains
    logger.debug("Evaluating hostname against approved domains: %s", APPROVED_API_DOMAINS)
    is_allowed = False
    for domain in APPROVED_API_DOMAINS:
        d = domain.lower()
        if hostname == d or hostname.endswith("." + d):
            logger.debug("Approved domain matched: %s", d)
            is_allowed = True
            break

    if not is_allowed:
        logger.debug("Validation failure: Hostname is not within approved domains.")
        raise ValueError("Domain is not approved for webhook calls.")

    logger.debug("Validation successful for webhook URL: %s", sanitized)
    return webhook_url


def process_webhook(webhook_url: str) -> str:
    """
    Process a webhook URL by validating that it targets an approved external API endpoint.

    This function serves as the public entry point for webhook URL handling. It:
      - Logs the received URL (sanitized; credentials, if any, removed).
      - Delegates validation to the helper function.
      - Logs the outcome (success or failure) for traceability.

    Note:
      - The returned value is intended to be used for internal API requests.
      - Configure logging in the host application to see INFO/DEBUG logs (e.g., logging.basicConfig).

    Args:
        webhook_url: The URL provided by an external service.

    Returns:
        A validated URL string that points to an approved API endpoint.

    Raises:
        ValueError: Propagated from validation if the URL is invalid or not approved.
    """
    sanitized = _sanitize_url_for_logging(webhook_url)
    logger.info("Received webhook URL for processing: %s", sanitized)

    try:
        validated = validate_webhook_url(webhook_url)
        logger.info("Webhook URL validated successfully: %s", sanitized)
        return validated

    except ValueError as e:
        logger.warning("Webhook URL validation failed: %s | reason=%s", sanitized, str(e))
        raise
