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
    if not callback_url:
        raise ValueError("Callback URL cannot be empty.")

    try:
        parsed_url = urlparse(callback_url)
    except Exception as e:
        raise ValueError(f"Invalid URL format: {e}")

    if not parsed_url.scheme or parsed_url.scheme.lower() not in ['http', 'https']:
        raise ValueError("URL scheme must be HTTP or HTTPS.")

    hostname = parsed_url.hostname
    if not hostname:
        raise ValueError("URL must have a valid hostname.")

    # Check if hostname is an internal/private address
    if hostname.lower() in [ip.lower() for ip in INTERNAL_IP_RANGES if not ip.endswith('.')]: # Exact matches like "localhost"
        raise ValueError(f"Access to internal/private network address '{hostname}' is forbidden.")
    
    for internal_range in [ip for ip in INTERNAL_IP_RANGES if ip.endswith('.')]: # Prefix matches like "10."
        if hostname.startswith(internal_range):
            raise ValueError(f"Access to internal/private network address range '{internal_range}*' for '{hostname}' is forbidden.")

    # Check if the domain is approved
    if hostname.lower() not in [domain.lower() for domain in APPROVED_API_DOMAINS]:
        raise ValueError(f"Domain '{hostname}' is not an approved API endpoint.")

    return callback_url
