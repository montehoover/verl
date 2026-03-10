import logging
from urllib.parse import urlparse

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]


def _get_hostname_from_url(callback_url: str) -> str:
    """Parses the URL and returns the hostname, raising ValueError if not found."""
    parsed_url = urlparse(callback_url)
    hostname = parsed_url.hostname
    if not hostname:
        raise ValueError("Invalid URL: Could not parse hostname.")
    return hostname


def _is_hostname_internal_or_localhost(hostname: str) -> bool:
    """Checks if the hostname matches any internal IP ranges or localhost."""
    return any(hostname.startswith(internal_ip) for internal_ip in INTERNAL_IP_RANGES)


def _is_hostname_approved(hostname: str) -> bool:
    """Checks if the hostname is in APPROVED_API_DOMAINS or a subdomain of an approved domain."""
    if hostname in APPROVED_API_DOMAINS:
        return True
    return any(hostname.endswith(f".{approved_domain}") for approved_domain in APPROVED_API_DOMAINS)


def _is_scheme_https(parsed_url) -> bool:
    """Checks if the URL scheme is HTTPS."""
    return parsed_url.scheme == "https"


def validate_callback(callback_url: str) -> str:
    """
    Validates a webhook URL from an external source.

    Args:
        callback_url: A string representing the external webhook URL.

    Returns:
        A validated and secure URL that is allowed for internal API interactions.

    Raises:
        ValueError: If the URL domain is not from the approved API endpoint list
                    or if it tries to access internal/private network addresses.
    """
    logger.info(f"Validating callback URL: {callback_url}")
    try:
        hostname = _get_hostname_from_url(callback_url)

        if _is_hostname_internal_or_localhost(hostname):
            error_message = f"Access to internal/private network address '{hostname}' is forbidden."
            logger.warning(f"Validation failed for URL '{callback_url}': {error_message}")
            raise ValueError(f"Invalid URL: {error_message}")

        if not _is_hostname_approved(hostname):
            error_message = f"Domain '{hostname}' is not an approved API endpoint."
            logger.warning(f"Validation failed for URL '{callback_url}': {error_message}")
            raise ValueError(f"Invalid URL: {error_message}")

        parsed_url = urlparse(callback_url)
        if not _is_scheme_https(parsed_url):
            error_message = "Scheme must be HTTPS."
            logger.warning(f"Validation failed for URL '{callback_url}': {error_message}")
            raise ValueError(f"Invalid URL: {error_message}")

        logger.info(f"Successfully validated callback URL: {callback_url}")
        return callback_url
    except ValueError as e:
        # Log the exception if it's raised by _get_hostname_from_url or re-raised
        if "Could not parse hostname" in str(e): # Check if it's the specific error from _get_hostname_from_url
             logger.error(f"Validation failed for URL '{callback_url}': {e}")
        # For other ValueErrors already logged, we just re-raise
        raise
