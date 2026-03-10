from urllib.parse import urlparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]


def _is_internal_network_location(hostname: str) -> bool:
    """Checks if the hostname points to an internal network location."""
    for internal_range in INTERNAL_IP_RANGES:
        if hostname == internal_range or \
           (internal_range.endswith('.') and hostname.startswith(internal_range)):
            return True
    return False


def _is_domain_approved(hostname: str) -> bool:
    """Checks if the hostname is in the list of approved API domains."""
    for approved_domain in APPROVED_API_DOMAINS:
        if hostname == approved_domain or hostname.endswith(f".{approved_domain}"):
            return True
    return False


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

    if _is_internal_network_location(hostname):
        raise ValueError(f"URL targets an internal network location: {hostname}")

    if not _is_domain_approved(hostname):
        raise ValueError(f"URL domain is not approved: {hostname}")

    return hook_url
