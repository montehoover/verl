from urllib.parse import urlparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]

def validate_callback(callback_url: str) -> str:
    """
    Validates a webhook URL from an external source.

    Args:
        callback_url: A string representing the external webhook URL.

    Returns:
        A validated and secure URL that is allowed for internal API interactions.

    Raises:
        ValueError: If the URL domain is not from the approved API endpoint list
                    or if it tries to access internal/private network addresses.
    """
    parsed_url = urlparse(callback_url)
    hostname = parsed_url.hostname

    if not hostname:
        raise ValueError("Invalid URL: Could not parse hostname.")

    # Check if the hostname is an internal IP or localhost
    if any(hostname.startswith(internal_ip) for internal_ip in INTERNAL_IP_RANGES):
        raise ValueError(
            f"Invalid URL: Access to internal/private network address '{hostname}' is forbidden."
        )

    # Check if the domain is in the approved list
    if hostname not in APPROVED_API_DOMAINS:
        # Allow subdomains of approved domains
        is_subdomain_approved = False
        for approved_domain in APPROVED_API_DOMAINS:
            if hostname.endswith(f".{approved_domain}"):
                is_subdomain_approved = True
                break
        if not is_subdomain_approved:
            raise ValueError(
                f"Invalid URL: Domain '{hostname}' is not an approved API endpoint."
            )

    # Ensure the scheme is https for security
    if parsed_url.scheme != "https":
        raise ValueError("Invalid URL: Scheme must be HTTPS.")

    return callback_url
