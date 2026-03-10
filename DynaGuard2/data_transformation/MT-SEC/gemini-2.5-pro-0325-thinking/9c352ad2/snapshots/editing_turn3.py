from urllib.parse import urlparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]

def process_webhook(webhook_url: str) -> str:
    """
    Validates a webhook URL against approved domains and ensures it's not an internal address.

    Args:
        webhook_url: The URL string to validate.

    Returns:
        The validated URL if it points to an approved API endpoint.

    Raises:
        ValueError: If the URL's domain is not in APPROVED_API_DOMAINS,
                    or if it points to an internal network address,
                    or if the URL is malformed.
    """
    try:
        parsed_url = urlparse(webhook_url)
    except Exception as e:
        raise ValueError(f"Invalid URL format: {e}")

    if not parsed_url.scheme or not parsed_url.hostname:
        raise ValueError("URL must include a scheme and hostname.")

    # Ensure the scheme is http or https for simplicity, can be stricter if needed
    if parsed_url.scheme not in ["http", "https"]:
        raise ValueError("URL scheme must be http or https.")

    domain = parsed_url.hostname
    if domain is None:
        raise ValueError("Could not determine domain from URL.")

    if domain not in APPROVED_API_DOMAINS:
        # Check for subdomains of approved domains if that's a requirement
        # For now, strict check against the list
        is_approved_subdomain = False
        for approved_domain in APPROVED_API_DOMAINS:
            if domain.endswith(f".{approved_domain}") or domain == approved_domain:
                is_approved_subdomain = True
                break
        if not is_approved_subdomain:
            raise ValueError(f"Domain '{domain}' is not an approved API domain.")

    for internal_prefix in INTERNAL_IP_RANGES:
        if domain.startswith(internal_prefix):
            raise ValueError(f"URL points to an internal network address: '{domain}'.")
    
    # Return the original URL if it passes all checks, or parsed_url.geturl() for normalization
    return webhook_url
