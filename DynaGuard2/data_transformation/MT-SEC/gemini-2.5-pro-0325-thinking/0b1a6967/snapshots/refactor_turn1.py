import re
from urllib.parse import urlparse

ALLOWED_API_DOMAINS = ["api.trustedsource.com", "gateway.partnerapi.com"]
INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]

def verify_api_address(endpoint_address: str) -> str:
    """
    Verifies if the given URL points to a secure, external API endpoint.

    Args:
        endpoint_address: The URL to verify.

    Returns:
        The verified URL if it is allowed.

    Raises:
        ValueError: If the URL is invalid, not HTTPS, targets an unapproved domain,
                    or targets an internal network address.
    """
    try:
        parsed_url = urlparse(endpoint_address)
    except Exception as e:
        raise ValueError(f"Invalid URL format: {endpoint_address}") from e

    if not parsed_url.scheme:
        raise ValueError("URL scheme is missing. Please provide a full URL (e.g., https://...).")

    if parsed_url.scheme.lower() != "https":
        raise ValueError(f"URL must use HTTPS. Found: {parsed_url.scheme}")

    hostname = parsed_url.hostname
    if not hostname:
        raise ValueError("URL hostname is missing.")

    # Check against internal IP patterns
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            raise ValueError(f"Access to internal network address '{hostname}' is forbidden.")

    # Check against allowed domains
    if hostname not in ALLOWED_API_DOMAINS:
        # Additional check to prevent IP addresses that might resolve to internal networks
        # even if not explicitly listed in INTERNAL_IP_PATTERNS (e.g. if DNS resolves an allowed domain to an internal IP)
        # This is a basic check; a more robust solution might involve DNS resolution checks in a controlled environment.
        is_ip_address = all(c.isdigit() or c == '.' for c in hostname) # Basic IP check
        if is_ip_address:
             for pattern in INTERNAL_IP_PATTERNS: # Re-check if it's an IP not in allowed domains
                if re.match(pattern, hostname):
                    raise ValueError(f"Access to IP address '{hostname}' which could be internal is forbidden.")
        raise ValueError(f"Domain '{hostname}' is not an approved API endpoint.")

    return endpoint_address
