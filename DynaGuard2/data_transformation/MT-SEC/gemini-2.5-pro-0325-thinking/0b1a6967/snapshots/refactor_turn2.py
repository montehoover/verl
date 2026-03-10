import re
from urllib.parse import urlparse

ALLOWED_API_DOMAINS = ["api.trustedsource.com", "gateway.partnerapi.com"]
INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]


def _is_internal_ip(hostname: str) -> bool:
    """Checks if the hostname matches any internal IP patterns."""
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            return True
    return False


def _is_allowed_domain(hostname: str) -> bool:
    """Checks if the hostname is in the list of allowed API domains."""
    return hostname in ALLOWED_API_DOMAINS


def verify_api_address(endpoint_address: str) -> str:
    """
    Verifies if the given URL points to a secure, external API endpoint.
    Uses a pipeline of validation checks:
    1. Parse URL.
    2. Validate scheme (must be HTTPS).
    3. Validate hostname presence.
    4. Check if hostname matches internal IP patterns.
    5. Check if hostname is in allowed domains.
    6. If validation passes, return the URL; otherwise, raise ValueError.

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

    # Validation Step 1: Scheme check
    if not parsed_url.scheme:
        raise ValueError("URL scheme is missing. Please provide a full URL (e.g., https://...).")
    if parsed_url.scheme.lower() != "https":
        raise ValueError(f"URL must use HTTPS. Found: {parsed_url.scheme}")

    # Validation Step 2: Hostname presence
    hostname = parsed_url.hostname
    if not hostname:
        raise ValueError("URL hostname is missing.")

    # Validation Step 3: Internal IP check
    # This check applies to the hostname string.
    if _is_internal_ip(hostname):
        raise ValueError(f"Access to internal network address '{hostname}' is forbidden.")

    # Validation Step 4: Allowed domain check
    if _is_allowed_domain(hostname):
        return endpoint_address  # Approved domain

    # Validation Step 5: Handle non-allowed hostnames
    # If the hostname is not an internal IP (checked above) and not an allowed domain,
    # it's an unapproved endpoint. This matches the original logic's final rejection.
    raise ValueError(f"Domain '{hostname}' is not an approved API endpoint.")
