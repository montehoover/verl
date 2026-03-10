import os
import logging
from urllib.parse import urlparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]

# Configure logger to write to a file in the current directory
LOG_FILE = os.path.join(os.getcwd(), "webhook_analysis.log")
logger = logging.getLogger("webhook_analyzer")
logger.setLevel(logging.INFO)
logger.propagate = False
# Ensure we don't add duplicate handlers if this module is imported multiple times
_log_file_abs = os.path.abspath(LOG_FILE)
if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == _log_file_abs for h in logger.handlers):
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def is_internal_network_host(hostname: str, ranges=INTERNAL_IP_RANGES) -> bool:
    """
    Determine if the given hostname points to an internal/private network.

    The check is based on exact or prefix matches against provided ranges,
    such as "localhost", "127.0.0.1", "10.", "192.168.".

    Args:
        hostname: The normalized hostname to check.
        ranges: A list of exact or prefix patterns representing internal networks.

    Returns:
        True if the hostname matches an internal/private pattern, otherwise False.
    """
    for pattern in ranges:
        if hostname == pattern or hostname.startswith(pattern):
            return True
    return False


def is_approved_api_domain(hostname: str, approved=APPROVED_API_DOMAINS) -> bool:
    """
    Check if the given hostname is an approved API domain or its subdomain.

    Args:
        hostname: The normalized hostname to check.
        approved: A list of approved apex domains.

    Returns:
        True if hostname is exactly an approved domain or a subdomain of one, otherwise False.
    """
    return any(hostname == domain or hostname.endswith("." + domain) for domain in approved)


def analyze_webhook(webhook_address: str) -> str:
    """
    Validate a webhook URL from an external service and return a URL that can be safely used
    for internal API requests.

    Args:
        webhook_address: A string representing the URL supplied by an external service.

    Returns:
        A validated URL string.

    Raises:
        ValueError: If the URL is invalid, the domain is not approved, or the URL points to
                    an internal/private network.
    """
    logger.info("Received webhook URL: %s", webhook_address)

    if not isinstance(webhook_address, str) or not webhook_address.strip():
        logger.warning("Decision: REJECT - invalid URL input (non-empty string required)")
        raise ValueError("Invalid URL: must be a non-empty string")

    parsed = urlparse(webhook_address.strip())
    logger.info("Validation step: parsed URL - scheme=%s, netloc=%s", parsed.scheme, parsed.netloc)

    # Require a valid scheme and network location
    if parsed.scheme.lower() not in ("http", "https"):
        logger.warning("Decision: REJECT - unsupported scheme: %s", parsed.scheme)
        raise ValueError("Invalid URL: unsupported scheme")
    if not parsed.netloc:
        logger.warning("Decision: REJECT - missing host/netloc")
        raise ValueError("Invalid URL: missing host")

    # Extract and normalize the hostname
    hostname = (parsed.hostname or "").rstrip(".").lower()
    logger.info("Validation step: normalized hostname - %s", hostname or "<empty>")

    if not hostname:
        logger.warning("Decision: REJECT - missing hostname after parsing")
        raise ValueError("Invalid URL: missing hostname")

    # Block internal/private network targets
    logger.info("Validation step: checking for internal/private network")
    if is_internal_network_host(hostname):
        logger.warning("Decision: REJECT - internal/private network host: %s", hostname)
        raise ValueError("Rejected URL: points to an internal/private network")

    # Ensure the hostname belongs to an approved domain (exact match or subdomain)
    logger.info("Validation step: checking against approved API domains")
    if not is_approved_api_domain(hostname):
        logger.warning("Decision: REJECT - unapproved domain: %s", hostname)
        raise ValueError("Rejected URL: domain not in approved API list")

    # Passed validation; return the original URL
    logger.info("Decision: ACCEPT - URL approved for internal API request")
    return webhook_address
