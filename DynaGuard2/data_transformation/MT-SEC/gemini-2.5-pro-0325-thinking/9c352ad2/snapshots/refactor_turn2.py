from urllib.parse import urlparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]


def _validate_webhook_url_domain(hostname: str) -> None:
    """
    Validates if the hostname belongs to approved domains and is not an internal address.

    Args:
        hostname: The hostname to validate.

    Raises:
        ValueError: If the hostname is an internal address or not an approved domain.
    """
    # Check if the hostname is an internal IP or localhost
    for internal_range in INTERNAL_IP_RANGES:
        if hostname.startswith(internal_range):
            raise ValueError(
                f"Invalid URL: Access to internal address '{hostname}' is not allowed."
            )

    # Check if the domain is approved
    if hostname not in APPROVED_API_DOMAINS:
        raise ValueError(
            f"Invalid URL: Domain '{hostname}' is not an approved API endpoint."
        )


def process_webhook(webhook_url: str) -> str:
    """
    Processes a webhook URL provided by an external service.
    The returned URL will be used to perform an internal API request.

    Args:
        webhook_url: The URL provided by an external service.

    Returns:
        A validated URL that points to an approved API endpoint.

    Raises:
        ValueError: If the URL's domain does not belong to pre-approved
                    API endpoints or an internal address is attempted.
    """
    parsed_url = urlparse(webhook_url)
    hostname = parsed_url.hostname

    if not hostname:
        raise ValueError("Invalid URL: Could not parse hostname.")

    # Validate the extracted hostname
    _validate_webhook_url_domain(hostname)

    return webhook_url
