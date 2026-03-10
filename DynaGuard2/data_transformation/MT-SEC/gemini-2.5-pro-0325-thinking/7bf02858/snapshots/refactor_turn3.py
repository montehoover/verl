import logging
from urllib.parse import urlparse

# Basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]


def _is_internal_network_location(hostname: str) -> bool:
    """Checks if the hostname points to an internal network location."""
    for internal_range in INTERNAL_IP_RANGES:
        if hostname == internal_range or \
           (internal_range.endswith('.') and hostname.startswith(internal_range)):
            return True
    return False


def _is_domain_approved(hostname: str) -> bool:
    """Checks if the hostname is in the list of approved API domains."""
    for approved_domain in APPROVED_API_DOMAINS:
        if hostname == approved_domain or hostname.endswith(f".{approved_domain}"):
            return True
    return False


def process_hook(hook_url: str) -> str:
    """
    Handles a webhook URL sent by an external service. The resulting URL will be
    used to invoke an internal API action.

    Args:
        hook_url: A string representing the URL provided by the external service.

    Returns:
        A safely validated URL that references an approved API endpoint.

    Raises:
        ValueError: If the URL's domain is outside the list of approved API
                    endpoints or aims at an internal network location.
    """
    logger.info(f"Processing hook_url: {hook_url}")
    parsed_url = urlparse(hook_url)
    hostname = parsed_url.hostname

    if not hostname:
        logger.error(f"Invalid URL: Could not parse hostname from {hook_url}")
        raise ValueError("Invalid URL: Could not parse hostname.")

    logger.info(f"Validating hostname: {hostname}")

    if _is_internal_network_location(hostname):
        logger.warning(f"Validation failed: {hostname} targets an internal network location.")
        raise ValueError(f"URL targets an internal network location: {hostname}")
    logger.info(f"Hostname {hostname} is not an internal network location.")

    if not _is_domain_approved(hostname):
        logger.warning(f"Validation failed: {hostname} domain is not approved.")
        raise ValueError(f"URL domain is not approved: {hostname}")
    logger.info(f"Hostname {hostname} domain is approved.")

    logger.info(f"Successfully validated hook_url: {hook_url}")
    return hook_url
