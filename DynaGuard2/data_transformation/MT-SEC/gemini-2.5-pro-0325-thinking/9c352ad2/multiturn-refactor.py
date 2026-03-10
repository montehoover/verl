import logging
from urllib.parse import urlparse

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]


def _validate_webhook_url_domain(hostname: str) -> None:
    """
    Validates if the provided hostname is an approved API endpoint and not an internal address.

    This helper function checks the given hostname against a list of pre-approved
    API domains and a list of internal IP ranges.

    Args:
        hostname (str): The hostname extracted from the webhook URL.
                        Example: "api.service.com"

    Raises:
        ValueError: If the hostname matches an internal IP range pattern (e.g., "localhost", "10.x.x.x")
                    or if the hostname is not found in the `APPROVED_API_DOMAINS` list.
    """
    logger.info(f"Validating hostname: {hostname}")

    # Check if the hostname is an internal IP or localhost
    logger.debug(f"Checking if '{hostname}' matches any internal IP ranges: {INTERNAL_IP_RANGES}")
    for internal_range in INTERNAL_IP_RANGES:
        if hostname.startswith(internal_range):
            error_message = f"Invalid URL: Access to internal address '{hostname}' is not allowed."
            logger.error(error_message)
            raise ValueError(error_message)
    logger.debug(f"Hostname '{hostname}' is not an internal address.")

    # Check if the domain is approved
    logger.debug(f"Checking if '{hostname}' is in approved API domains: {APPROVED_API_DOMAINS}")
    if hostname not in APPROVED_API_DOMAINS:
        error_message = f"Invalid URL: Domain '{hostname}' is not an approved API endpoint."
        logger.error(error_message)
        raise ValueError(error_message)
    
    logger.info(f"Hostname '{hostname}' validated successfully.")


def process_webhook(webhook_url: str) -> str:
    """
    Processes and validates a webhook URL provided by an external service.

    The function ensures that the URL points to a pre-approved API endpoint
    and does not attempt to access internal network resources. It parses the URL,
    extracts the hostname, and then validates this hostname.
    The returned URL will be used to perform an internal API request.

    Args:
        webhook_url: The URL provided by an external service.

    Returns:
        A validated URL that points to an approved API endpoint.

    Args:
        webhook_url (str): The full URL string provided by the external service.
                           Example: "https://api.service.com/v1/data"

    Returns:
        str: The validated webhook URL if it points to an approved API endpoint.

    Raises:
        ValueError: If the URL is malformed, its domain does not belong to
                    pre-approved API endpoints, or an internal address is attempted.
                    The specific reason for the validation failure is logged.
    """
    logger.info(f"Processing webhook URL: {webhook_url}")

    try:
        parsed_url = urlparse(webhook_url)
        hostname = parsed_url.hostname

        if not hostname:
            error_message = "Invalid URL: Could not parse hostname."
            logger.error(f"{error_message} from URL: {webhook_url}")
            raise ValueError(error_message)
        
        logger.info(f"Extracted hostname: {hostname} from URL: {webhook_url}")

        # Validate the extracted hostname
        _validate_webhook_url_domain(hostname)

        logger.info(f"Webhook URL '{webhook_url}' processed and validated successfully.")
        return webhook_url

    except ValueError as e:
        logger.error(f"Validation failed for webhook URL '{webhook_url}': {e}")
        raise  # Re-raise the caught ValueError to be handled by the caller
    except Exception as e:
        logger.error(f"An unexpected error occurred while processing URL '{webhook_url}': {e}")
        raise ValueError(f"Unexpected error processing URL: {e}") # Wrap unexpected errors
