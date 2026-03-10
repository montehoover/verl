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
    Validate that the given URL points to an external trusted API endpoint.

    Args:
        api_url: The URL to be validated.

    Returns:
        The validated URL string.

    Raises:
        ValueError: If the URL is not HTTPS, missing a hostname, not part of the
                    allowed API domains, or attempts to reach internal addresses.
    """
    if not isinstance(api_url, str) or not api_url.strip():
        raise ValueError("A non-empty URL string is required.")

    parsed = urlparse(api_url)

    # Enforce HTTPS for external API calls.
    if parsed.scheme.lower() != "https":
        raise ValueError("Only HTTPS URLs are allowed for API requests.")

    if not parsed.netloc:
        raise ValueError("URL must include a network location (hostname).")

    host = parsed.hostname
    if not host:
        raise ValueError("URL contains an invalid or missing hostname.")

    # Block localhost and private/internal ranges using provided patterns.
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, host):
            raise ValueError("URL points to an internal network address, which is not allowed.")

    # Ensure the host is exactly an allowed domain or a subdomain of one.
    hostname = host.lower().strip(".")
    is_allowed_domain = False
    for allowed in ALLOWED_API_DOMAINS:
        allowed = allowed.lower().strip(".")
        if hostname == allowed or hostname.endswith("." + allowed):
            is_allowed_domain = True
            break

    if not is_allowed_domain:
        raise ValueError("URL is not part of the allowed API domains.")

    return api_url
