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

    # Check if the domain is approved
    if hostname not in APPROVED_API_DOMAINS:
        # Check subdomains for approved domains if applicable
        # For example, if *.api.service.com is allowed, then
        # any subdomain of api.service.com would be valid.
        # Current implementation requires exact match in APPROVED_API_DOMAINS.
        # If subdomains are allowed, the logic below needs adjustment.
        # For now, we assume exact domain match.
        is_approved_subdomain = False
        for approved_domain in APPROVED_API_DOMAINS:
            if hostname.endswith(f".{approved_domain}"):
                 # This logic might need refinement based on how subdomains are approved.
                 # For example, if "api.service.com" is approved, is "sub.api.service.com" approved?
                 # The current APPROVED_API_DOMAINS list suggests exact matches.
                 # If "service.com" was in the list and we wanted to allow subdomains,
                 # then this check would be more relevant.
                 # Given the current list, this check might be too broad or not needed.
                 # For simplicity and strictness based on current APPROVED_API_DOMAINS:
                 pass # Keeping it simple, direct match is checked below.

        if not any(hostname == domain for domain in APPROVED_API_DOMAINS):
             raise ValueError(
                f"Invalid URL: Domain '{hostname}' is not an approved API endpoint."
            )

    return webhook_url
