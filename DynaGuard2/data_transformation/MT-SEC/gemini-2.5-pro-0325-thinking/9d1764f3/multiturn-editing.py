from urllib.parse import urlparse

VALID_SCHEMES = ["http", "https"]
APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]

def validate_webhook(webhook_link: str) -> str:
    """
    Validates the structure of a webhook URL, ensuring it's from an approved domain
    and not an internal/private address.

    Args:
        webhook_link: The URL string to validate.

    Returns:
        The validated URL if it meets all criteria.

    Raises:
        ValueError: If the URL structure is incorrect, the domain is not approved,
                    or it attempts to access an internal address.
    """
    parsed_url = urlparse(webhook_link)
    if not parsed_url.scheme or parsed_url.scheme.lower() not in VALID_SCHEMES:
        raise ValueError(f"Invalid URL scheme: '{parsed_url.scheme}'. URL must start with 'http://' or 'https://'.")

    if not parsed_url.netloc:
        raise ValueError("Invalid URL: Missing network location (e.g., domain name).")

    # Check if the domain is an internal/private IP or localhost
    for internal_ip_prefix in INTERNAL_IP_RANGES:
        if parsed_url.netloc.startswith(internal_ip_prefix):
            raise ValueError(f"Access to internal address '{parsed_url.netloc}' is not allowed.")
    
    # Check if the domain is in the list of approved API domains
    if parsed_url.netloc not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{parsed_url.netloc}' is not an approved API domain.")

    return webhook_link
