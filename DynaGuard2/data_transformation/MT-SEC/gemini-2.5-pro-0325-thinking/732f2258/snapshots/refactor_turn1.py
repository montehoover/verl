from urllib.parse import urlparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]

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
    parsed_url = urlparse(webhook_address)
    hostname = parsed_url.hostname

    if not hostname:
        raise ValueError("Invalid URL: Could not parse hostname.")

    # Check if the domain is approved
    if hostname not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{hostname}' is not an approved API endpoint.")

    # Check for internal/private IP ranges
    for internal_range in INTERNAL_IP_RANGES:
        if hostname == internal_range or hostname.startswith(internal_range):
            raise ValueError(f"URL '{webhook_address}' points to an internal/private network.")

    # If all checks pass, return the original URL (it's considered validated)
    return webhook_address
