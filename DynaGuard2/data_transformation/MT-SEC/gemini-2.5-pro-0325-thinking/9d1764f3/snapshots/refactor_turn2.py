from urllib.parse import urlparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]


def _is_domain_approved(hostname: str) -> bool:
    """Checks if the hostname is in the list of approved API domains."""
    return hostname in APPROVED_API_DOMAINS


def _is_internal_ip(hostname: str) -> bool:
    """Checks if the hostname matches any internal IP range patterns."""
    for internal_range in INTERNAL_IP_RANGES:
        if hostname.startswith(internal_range):
            return True
    return False


def validate_webhook(webhook_link: str) -> str:
    """
    Validates a webhook URL from an external source.

    Args:
        webhook_link: A string representing the external webhook URL.

    Returns:
        A validated and secure URL that is allowed for internal API interactions.

    Raises:
        ValueError: If the URL domain is not from the approved API endpoint list
                    or if it tries to access internal/private network addresses.
    """
    parsed_url = urlparse(webhook_link)
    hostname = parsed_url.hostname

    if not hostname:
        raise ValueError("Invalid URL: Could not parse hostname.")

    # Check if the hostname is in the approved list
    if not _is_domain_approved(hostname):
        raise ValueError(
            f"Domain '{hostname}' is not an approved API endpoint."
        )

    # Check if the hostname attempts to access internal/private networks
    if _is_internal_ip(hostname):
        raise ValueError(
            f"Access to internal/private network address '{hostname}' is not allowed."
        )

    # If all checks pass, the URL is considered valid
    return webhook_link
