import logging
from urllib.parse import urlparse

# Configure logging (basic configuration, can be customized further)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]


def _validate_domain_approved(hostname: str):
    """Checks if the hostname is in the list of approved API domains."""
    if hostname not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{hostname}' is not an approved API endpoint.")


def _validate_no_internal_network(hostname: str, external_url: str):
    """Checks if the hostname points to an internal/private network."""
    for internal_range in INTERNAL_IP_RANGES:
        if hostname == internal_range or hostname.startswith(internal_range):
            raise ValueError(f"URL '{external_url}' points to an internal/private network.")


def handle_webhook_request(external_url: str) -> str:
    """
    Processes a webhook URL sent from an external service.

    This function validates the external URL to ensure it belongs to an approved
    domain and does not point to an internal or private network.

    Args:
        external_url: A string representing the URL supplied by an external service.

    Returns:
        A validated URL that can be safely used for internal API requests.

    Raises:
        ValueError: If the domain doesn't belong to the allowed API endpoint list
                    or if the URL points to an internal/private network.
    """
    logger.info(f"Processing webhook request for URL: {external_url}")

    try:
        parsed_url = urlparse(external_url)
        hostname = parsed_url.hostname

        if not hostname:
            logger.error(f"Invalid URL: Could not parse hostname from '{external_url}'.")
            raise ValueError("Invalid URL: Could not parse hostname.")

        logger.info(f"Validating domain for hostname: {hostname}")
        _validate_domain_approved(hostname)
        logger.info(f"Domain '{hostname}' is approved.")

        logger.info(f"Checking for internal network for hostname: {hostname}, URL: {external_url}")
        _validate_no_internal_network(hostname, external_url)
        logger.info(f"URL '{external_url}' does not point to an internal network.")

        logger.info(f"Webhook URL validation successful for: {external_url}")
        # If all checks pass, return the original URL (or a specifically formatted one if needed)
        # For now, returning the validated external_url as is.
        return external_url
    except ValueError as e:
        logger.error(f"Validation failed for URL '{external_url}': {e}")
        raise
