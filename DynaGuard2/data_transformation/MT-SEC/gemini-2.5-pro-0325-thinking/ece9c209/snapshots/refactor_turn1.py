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
    Ensures that the URL points to an external trusted API endpoint.

    Args:
        api_url: The URL to be validated.

    Returns:
        The validated URL.

    Raises:
        ValueError: If the URL is not part of the allowed API domains
                    or attempts to reach internal network addresses.
    """
    try:
        parsed_url = urlparse(api_url)
        hostname = parsed_url.hostname
    except Exception as e:
        raise ValueError(f"Invalid URL: {api_url}. Error: {e}")

    if not hostname:
        raise ValueError(f"Invalid URL: {api_url}. Could not parse hostname.")

    # Check against internal IP patterns
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            raise ValueError(
                f"Access to internal network address '{hostname}' is forbidden."
            )

    # Check against allowed domains
    if hostname not in ALLOWED_API_DOMAINS:
        # Check if it's a subdomain of an allowed domain
        is_allowed_subdomain = False
        for domain in ALLOWED_API_DOMAINS:
            if hostname.endswith(f".{domain}"):
                is_allowed_subdomain = True
                break
        if not is_allowed_subdomain:
            raise ValueError(
                f"Domain '{hostname}' is not an allowed API domain."
            )

    return api_url
