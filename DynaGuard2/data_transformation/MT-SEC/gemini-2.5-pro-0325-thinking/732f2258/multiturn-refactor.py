import logging
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(
    filename='webhook_analyzer.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]


def _is_domain_approved(hostname: str) -> bool:
    """Checks if the hostname is in the list of approved API domains."""
    return hostname in APPROVED_API_DOMAINS


def _is_internal_network(hostname: str) -> bool:
    """Checks if the hostname points to an internal/private network."""
    for internal_range in INTERNAL_IP_RANGES:
        if hostname == internal_range or hostname.startswith(internal_range):
            return True
    return False


def analyze_webhook(webhook_address: str) -> str:
    """
    Processes a webhook URL sent from an external service and returns a
    validated URL for internal API requests.

    Args:
        webhook_address: A string representing the URL supplied by an
                         external service.

    Returns:
        A validated URL that can be safely used for internal API requests.

    Raises:
        ValueError: If the domain doesn't belong to the allowed API
                    endpoint list or if the URL points to an
                    internal/private network.
    """
    logging.info(f"Analyzing webhook URL: {webhook_address}")
    parsed_url = urlparse(webhook_address)
    hostname = parsed_url.hostname

    if not hostname:
        logging.error(f"Invalid URL: Could not parse hostname from '{webhook_address}'.")
        raise ValueError("Invalid URL: Could not parse hostname.")
    logging.info(f"Parsed hostname: {hostname} from URL: {webhook_address}")

    # Check if the domain is approved
    if not _is_domain_approved(hostname):
        logging.warning(f"Domain validation failed: '{hostname}' is not an approved API endpoint. URL: {webhook_address}")
        raise ValueError(f"Domain '{hostname}' is not an approved API endpoint.")
    logging.info(f"Domain validation successful for hostname: {hostname}. URL: {webhook_address}")

    # Check for internal/private IP ranges
    if _is_internal_network(hostname):
        logging.warning(f"Internal network check failed: URL '{webhook_address}' (hostname: {hostname}) points to an internal/private network.")
        raise ValueError(f"URL '{webhook_address}' points to an internal/private network.")
    logging.info(f"Internal network check successful for hostname: {hostname}. URL: {webhook_address}")

    # If all checks pass, return the original URL (it's considered validated)
    logging.info(f"Webhook URL successfully validated: {webhook_address}")
    return webhook_address
