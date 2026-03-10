import re
import logging
from urllib.parse import urlparse

# Configure basic logging
# In a real application, this would likely be configured globally.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ALLOWED_API_DOMAINS = ["api.trustedsource.com", "gateway.partnerapi.com"]
INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]


def _is_internal_ip(hostname: str) -> bool:
    """Checks if the hostname matches any internal IP patterns."""
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            return True
    return False


def _is_allowed_domain(hostname: str) -> bool:
    """Checks if the hostname is in the list of allowed API domains."""
    return hostname in ALLOWED_API_DOMAINS


def verify_api_address(endpoint_address: str) -> str:
    """
    Verifies if the given URL points to a secure, external API endpoint.
    Uses a pipeline of validation checks:
    1. Parse URL.
    2. Validate scheme (must be HTTPS).
    3. Validate hostname presence.
    4. Check if hostname matches internal IP patterns.
    5. Check if hostname is in allowed domains.
    6. If validation passes, return the URL; otherwise, raise ValueError.

    Args:
        endpoint_address: The URL to verify.

    Returns:
        The verified URL if it is allowed.

    Raises:
        ValueError: If the URL is invalid, not HTTPS, targets an unapproved domain,
                    or targets an internal network address.
    """
    logger.info(f"Attempting to verify API address: {endpoint_address}")

    try:
        parsed_url = urlparse(endpoint_address)
        logger.info(f"Successfully parsed URL: {endpoint_address}")
    except Exception as e:
        logger.error(f"Invalid URL format for {endpoint_address}: {e}")
        raise ValueError(f"Invalid URL format: {endpoint_address}") from e

    # Validation Step 1: Scheme check
    logger.info(f"Validating scheme for {endpoint_address}")
    if not parsed_url.scheme:
        logger.warning(f"Scheme validation failed for {endpoint_address}: Scheme is missing.")
        raise ValueError("URL scheme is missing. Please provide a full URL (e.g., https://...).")
    if parsed_url.scheme.lower() != "https":
        logger.warning(f"Scheme validation failed for {endpoint_address}: Must be HTTPS, found {parsed_url.scheme}.")
        raise ValueError(f"URL must use HTTPS. Found: {parsed_url.scheme}")
    logger.info(f"Scheme validation passed for {endpoint_address}")

    # Validation Step 2: Hostname presence
    hostname = parsed_url.hostname
    logger.info(f"Validating hostname presence for {endpoint_address}. Found hostname: {hostname}")
    if not hostname:
        logger.warning(f"Hostname validation failed for {endpoint_address}: Hostname is missing.")
        raise ValueError("URL hostname is missing.")
    logger.info(f"Hostname validation passed for {endpoint_address}")

    # Validation Step 3: Internal IP check
    logger.info(f"Checking if hostname '{hostname}' is an internal IP.")
    if _is_internal_ip(hostname):
        logger.warning(f"Internal IP check failed for {endpoint_address}: Hostname '{hostname}' is an internal IP.")
        raise ValueError(f"Access to internal network address '{hostname}' is forbidden.")
    logger.info(f"Hostname '{hostname}' is not an internal IP.")

    # Validation Step 4: Allowed domain check
    logger.info(f"Checking if hostname '{hostname}' is an allowed domain.")
    if _is_allowed_domain(hostname):
        logger.info(f"Domain validation passed for {endpoint_address}: Hostname '{hostname}' is an allowed domain.")
        logger.info(f"Successfully verified API address: {endpoint_address}")
        return endpoint_address  # Approved domain
    logger.info(f"Hostname '{hostname}' is not in the allowed domains list.")

    # Validation Step 5: Handle non-allowed hostnames
    # If the hostname is not an internal IP (checked above) and not an allowed domain,
    # it's an unapproved endpoint.
    logger.warning(f"Domain validation failed for {endpoint_address}: Hostname '{hostname}' is not an approved API endpoint.")
    raise ValueError(f"Domain '{hostname}' is not an approved API endpoint.")
