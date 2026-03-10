from urllib.parse import urlparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]

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
    parsed_url = urlparse(external_url)
    hostname = parsed_url.hostname

    if not hostname:
        raise ValueError("Invalid URL: Could not parse hostname.")

    # Check if the domain is approved
    if hostname not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{hostname}' is not an approved API endpoint.")

    # Check for internal/private IP addresses
    for internal_range in INTERNAL_IP_RANGES:
        if hostname == internal_range or hostname.startswith(internal_range):
            raise ValueError(f"URL '{external_url}' points to an internal/private network.")

    # If all checks pass, return the original URL (or a specifically formatted one if needed)
    # For now, returning the validated external_url as is.
    return external_url
