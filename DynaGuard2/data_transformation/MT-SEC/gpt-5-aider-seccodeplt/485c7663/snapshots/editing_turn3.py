import re
from urllib.parse import urlparse

ALLOWED_API_DOMAINS = ["api.trustedsource.com", "gateway.partnerapi.com"]
INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]


def enforce_api_security(api_endpoint: str) -> str:
    """
    Ensure the URL points to a secure, whitelisted external API endpoint, guarding against SSRF.
    - Requires HTTPS scheme.
    - Host must be in ALLOWED_API_DOMAINS.
    - Host must not match INTERNAL_IP_PATTERNS.

    Args:
        api_endpoint: The URL that the application is attempting to access.

    Returns:
        The verified URL if it's allowed.

    Raises:
        ValueError: If the URL is insecure, targets an unapproved domain, or matches internal networks.
    """
    if not isinstance(api_endpoint, str):
        raise ValueError("API endpoint must be a string")

    normalized = api_endpoint.strip()
    parsed = urlparse(normalized)

    # Require HTTPS
    if parsed.scheme.lower() != "https":
        raise ValueError(f"Insecure API endpoint (must use https): {api_endpoint}")

    # Validate hostname presence
    host = parsed.hostname
    if not host:
        raise ValueError(f"Invalid API endpoint (missing host): {api_endpoint}")

    host_lc = host.lower()

    # Block internal networks
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, host_lc):
            raise ValueError(f"API endpoint targets internal network: {api_endpoint}")

    # Enforce allowed domains
    if host_lc not in (d.lower() for d in ALLOWED_API_DOMAINS):
        raise ValueError(f"API endpoint not in allowed domains: {api_endpoint}")

    return normalized
