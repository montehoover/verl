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
        ValueError: If the URL is invalid, insecure, targets an unapproved domain,
                    or an internal network address.
    """
    try:
        parsed_url = urlparse(endpoint_address)
    except Exception as e:
        raise ValueError(f"Invalid URL format: {endpoint_address}. Error: {e}")

    if not parsed_url.scheme:
        raise ValueError(f"URL scheme is missing: {endpoint_address}")

    if parsed_url.scheme.lower() != "https":
        raise ValueError(f"URL must use HTTPS. Found: {parsed_url.scheme} in {endpoint_address}")

    hostname = parsed_url.hostname
    if not hostname:
        raise ValueError(f"Hostname could not be determined from URL: {endpoint_address}")

    # Check against internal IP patterns
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            raise ValueError(f"Access to internal or loopback address '{hostname}' is forbidden from URL: {endpoint_address}")

    # Check against allowed domains
    if hostname not in ALLOWED_API_DOMAINS:
        # Additional check to prevent subdomain bypass if primary domain is not allowed.
        # e.g. if "example.com" is allowed, "internal.example.com" might be if not explicitly checked.
        # However, current ALLOWED_API_DOMAINS are specific FQDNs.
        is_subdomain_of_allowed = False
        for allowed_domain in ALLOWED_API_DOMAINS:
            if hostname.endswith("." + allowed_domain): # Check if it's a subdomain of an allowed domain
                # This case should ideally not happen if ALLOWED_API_DOMAINS are strictly FQDNs
                # and not base domains. If base domains were allowed, this check would be more critical.
                # For now, with FQDNs, this means it's a different host.
                pass # Or raise error if subdomains of allowed domains are not permitted unless explicitly listed
        
        if not any(hostname == domain or hostname.endswith("." + domain) for domain in ALLOWED_API_DOMAINS):
             # Re-check to be absolutely sure, the previous check `hostname not in ALLOWED_API_DOMAINS` is sufficient
             # if ALLOWED_API_DOMAINS contains FQDNs.
             # This more complex check is for scenarios where ALLOWED_API_DOMAINS might contain base domains.
             # Given the current setup, `hostname not in ALLOWED_API_DOMAINS` is the primary gate.
            pass # Fall through to the final check

        if hostname not in ALLOWED_API_DOMAINS:
             raise ValueError(f"Domain '{hostname}' is not an approved API domain from URL: {endpoint_address}")


    return endpoint_address
