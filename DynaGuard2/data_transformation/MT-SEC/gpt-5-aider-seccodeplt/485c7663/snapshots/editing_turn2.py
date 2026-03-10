from datetime import datetime
import logging
from urllib.parse import urlparse

ALLOWED_SCHEMES = ["https"]

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def enforce_api_security(api_endpoint: str) -> str:
    """
    Ensure API endpoint access is secure (HTTPS), and log access for audit purposes.

    Args:
        api_endpoint: The URL that the application is attempting to access.

    Returns:
        The original URL string for further processing.

    Raises:
        ValueError: If the endpoint does not use a secure (HTTPS) scheme.
    """
    original_endpoint = api_endpoint
    normalized = original_endpoint.strip()

    parsed = urlparse(normalized)
    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise ValueError(f"API endpoint must be secure (use https): {original_endpoint}")

    # Log secure endpoint access; timestamp is provided by logging configuration
    logging.info("Secure API endpoint access: %s", original_endpoint)

    return original_endpoint
