from urllib.parse import urlparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]

def process_hook(hook_url: str) -> str:
    """
    Handles a webhook URL sent by an external service. The resulting URL will be
    used to invoke an internal API action.

    Args:
        hook_url: A string representing the URL provided by the external service.

    Returns:
        A safely validated URL that references an approved API endpoint.

    Raises:
        ValueError: If the URL's domain is outside the list of approved API
                    endpoints or aims at an internal network location.
    """
    parsed_url = urlparse(hook_url)
    hostname = parsed_url.hostname

    if not hostname:
        raise ValueError("Invalid URL: Could not parse hostname.")

    # Check if the hostname is an approved API domain
    if hostname not in APPROVED_API_DOMAINS:
        raise ValueError(
            f"Invalid URL: Domain '{hostname}' is not an approved API endpoint."
        )

    # Check if the hostname points to an internal network location
    for internal_range in INTERNAL_IP_RANGES:
        if hostname.startswith(internal_range):
            raise ValueError(
                f"Invalid URL: Domain '{hostname}' points to an internal network location."
            )

    return hook_url
