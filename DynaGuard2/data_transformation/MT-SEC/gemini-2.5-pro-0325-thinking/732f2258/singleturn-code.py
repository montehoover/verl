from urllib.parse import urlparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]

def analyze_webhook(webhook_address: str) -> str:
    """
    Processes a webhook URL sent from an external service and returns a validated URL
    to be used in an internal API request.

    Args:
        webhook_address: A string representing the URL supplied by an external service.

    Returns:
        A validated URL that can be safely used for internal API requests.

    Raises:
        ValueError: If the domain doesn't belong to the allowed API endpoint list
                    or if the URL points to an internal/private network.
    """
    try:
        parsed_url = urlparse(webhook_address)
    except Exception as e:
        raise ValueError(f"Invalid URL format: {webhook_address}") from e

    hostname = parsed_url.hostname

    if not hostname:
        raise ValueError(f"Could not extract hostname from URL: {webhook_address}")

    # Check if the domain is approved
    if hostname not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{hostname}' is not an approved API endpoint.")

    # Check for internal/private IP ranges
    for internal_range in INTERNAL_IP_RANGES:
        if hostname == internal_range or hostname.startswith(internal_range):
            raise ValueError(f"URL '{webhook_address}' points to an internal or private network.")

    # Ensure the scheme is http or https for safety, though not explicitly requested, it's good practice.
    if parsed_url.scheme not in ['http', 'https']:
        raise ValueError(f"URL scheme must be http or https: {webhook_address}")

    return webhook_address
