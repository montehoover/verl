from urllib.parse import urlparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]

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

    # Check if the hostname is an internal IP or localhost
    for internal_range in INTERNAL_IP_RANGES:
        if hostname.startswith(internal_range):
            raise ValueError(
                f"Invalid URL: Access to internal address '{hostname}' is not allowed."
            )

    # Check if the hostname is in the approved list
    if hostname not in APPROVED_API_DOMAINS:
        raise ValueError(
            f"Invalid URL: Domain '{hostname}' is not an approved API endpoint."
        )

    return webhook_url
