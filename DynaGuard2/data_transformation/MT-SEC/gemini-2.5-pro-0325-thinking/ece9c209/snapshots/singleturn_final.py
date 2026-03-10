import re
from urllib.parse import urlparse

ALLOWED_API_DOMAINS = ["api.trustedsource.com", "gateway.partnerapi.com"]
INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]

def validate_api_endpoint(api_url: str) -> str:
    """
    Ensures that the URL points to an external trusted API endpoint,
    preventing possible SSRF attacks.

    Args:
        api_url: The URL to be validated.

    Returns:
        A validated URL that points to a trusted API.

    Raises:
        ValueError: If the URL is not part of the allowed API domains,
                    attempts to reach internal network addresses, or has an invalid scheme.
    """
    try:
        parsed_url = urlparse(api_url)
    except Exception as e:
        raise ValueError(f"Invalid URL format: {api_url}. Error: {e}")

    if not parsed_url.scheme or parsed_url.scheme.lower() not in ['http', 'https']:
        raise ValueError(f"Invalid URL scheme: '{parsed_url.scheme}'. Only 'http' or 'https' are allowed.")

    hostname = parsed_url.hostname
    if not hostname:
        raise ValueError(f"URL must contain a valid hostname: {api_url}")

    # Check against internal IP patterns first
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            raise ValueError(
                f"URL hostname '{hostname}' matches an internal IP pattern. Access denied."
            )

    # Check against allowed domains
    if hostname not in ALLOWED_API_DOMAINS:
        # As a secondary check, ensure it's not an IP address that bypassed the internal IP check
        # (e.g. public IP not in allowed list)
        # This regex checks for a typical IPv4 format.
        if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", hostname):
             raise ValueError(
                f"URL hostname '{hostname}' is an IP address not in the allowed domains list. Access denied."
            )
        raise ValueError(
            f"URL hostname '{hostname}' is not in the allowed API domains. Access denied."
        )

    return api_url
