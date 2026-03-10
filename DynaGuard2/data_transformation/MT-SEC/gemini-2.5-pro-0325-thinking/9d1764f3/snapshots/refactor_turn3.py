import logging
from urllib.parse import urlparse

# Configure basic logging if not already configured by the application
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]


def _is_domain_approved(hostname: str) -> bool:
    """
    Checks if the provided hostname is present in the `APPROVED_API_DOMAINS` list.

    Args:
        hostname: The hostname string to validate.

    Returns:
        True if the hostname is in the approved list, False otherwise.
    """
    return hostname in APPROVED_API_DOMAINS


def _is_internal_ip(hostname: str) -> bool:
    """
    Checks if the provided hostname matches any of the patterns defined in
    `INTERNAL_IP_RANGES`, indicating it might be an internal or private IP address.

    Args:
        hostname: The hostname string to check.

    Returns:
        True if the hostname matches an internal IP pattern, False otherwise.
    """
    for internal_range in INTERNAL_IP_RANGES:
        if hostname.startswith(internal_range):
            return True
    return False


def validate_webhook(webhook_link: str) -> str:
    """
    Validates a webhook URL from an external source, ensuring it meets security
    and policy requirements before being used for internal API calls.

    It checks if the URL's domain is on an approved list and does not point to
    internal or private network addresses. Logging is performed at each step
    of the validation process.

    Args:
        webhook_link: A string representing the external webhook URL to be validated.

    Returns:
        The validated webhook_link if all checks pass, confirming it's a secure
        URL suitable for internal API interactions.

    Raises:
        ValueError: If the URL is malformed, its domain is not approved, or it
                    attempts to access an internal/private network address.
    """
    logger.info(f"Starting validation for webhook_link: {webhook_link}")
    parsed_url = urlparse(webhook_link)
    hostname = parsed_url.hostname

    if not hostname:
        logger.error(f"Invalid URL: Could not parse hostname from '{webhook_link}'.")
        raise ValueError("Invalid URL: Could not parse hostname.")
    logger.info(f"Successfully parsed hostname '{hostname}' from URL '{webhook_link}'.")

    # Check if the hostname is in the approved list
    if not _is_domain_approved(hostname):
        logger.warning(
            f"Validation failed: Domain '{hostname}' is not in APPROVED_API_DOMAINS."
        )
        raise ValueError(
            f"Domain '{hostname}' is not an approved API endpoint."
        )
    logger.info(f"Domain '{hostname}' is approved.")

    # Check if the hostname attempts to access internal/private networks
    if _is_internal_ip(hostname):
        logger.warning(
            f"Validation failed: Hostname '{hostname}' matches an internal IP range."
        )
        raise ValueError(
            f"Access to internal/private network address '{hostname}' is not allowed."
        )
    logger.info(f"Hostname '{hostname}' does not point to an internal IP range.")

    # If all checks pass, the URL is considered valid
    logger.info(f"Webhook URL '{webhook_link}' validated successfully.")
    return webhook_link
