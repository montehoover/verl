import re
import logging
from urllib.parse import urlparse

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ALLOWED_API_DOMAINS = ["api.trustedsource.com", "gateway.partnerapi.com"]
INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]


def _is_internal_ip(hostname: str) -> bool:
    """Checks if the hostname matches known internal IP patterns."""
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            return True
    return False


def _is_allowed_domain(hostname: str) -> bool:
    """Checks if the hostname is an allowed API domain or its subdomain."""
    if hostname in ALLOWED_API_DOMAINS:
        return True
    for domain in ALLOWED_API_DOMAINS:
        if hostname.endswith(f".{domain}"):
            return True
    return False


def validate_api_endpoint(api_url: str) -> str:
    """
    Ensures that the URL points to an external trusted API endpoint.

    Args:
        api_url: The URL to be validated.

    Returns:
        The validated URL.

    Raises:
        ValueError: If the URL is not part of the allowed API domains
                    or attempts to reach internal network addresses.
    """
    logger.info(f"Attempting to validate API endpoint: {api_url}")
    # Step 1: Parse the URL and extract the hostname.
    try:
        parsed_url = urlparse(api_url)
        hostname = parsed_url.hostname
    except Exception as e:
        # Raise an error if URL parsing fails.
        logger.error(f"Validation failed for URL '{api_url}': Invalid URL. Error: {e}")
        raise ValueError(f"Invalid URL: {api_url}. Error: {e}")

    if not hostname:
        # Raise an error if hostname cannot be extracted.
        logger.error(f"Validation failed for URL '{api_url}': Could not parse hostname.")
        raise ValueError(f"Invalid URL: {api_url}. Could not parse hostname.")

    # Step 2: Check if the hostname points to an internal IP address.
    if _is_internal_ip(hostname):
        # Raise an error if an internal IP address is detected.
        logger.error(
            f"Validation failed for URL '{api_url}': Access to internal network address '{hostname}' is forbidden."
        )
        raise ValueError(
            f"Access to internal network address '{hostname}' is forbidden."
        )

    # Step 3: Check if the hostname is an allowed API domain or its subdomain.
    if not _is_allowed_domain(hostname):
        # Raise an error if the domain is not in the allowed list.
        logger.error(
            f"Validation failed for URL '{api_url}': Domain '{hostname}' is not an allowed API domain."
        )
        raise ValueError(
            f"Domain '{hostname}' is not an allowed API domain."
        )

    # Step 4: If all checks pass, return the original URL.
    logger.info(f"Successfully validated API endpoint: {api_url}")
    return api_url
