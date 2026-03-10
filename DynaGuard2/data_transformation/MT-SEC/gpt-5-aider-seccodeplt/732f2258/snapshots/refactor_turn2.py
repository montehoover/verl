from urllib.parse import urlparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]


def is_internal_network_host(hostname: str, ranges=INTERNAL_IP_RANGES) -> bool:
    """
    Determine if the given hostname points to an internal/private network.

    The check is based on exact or prefix matches against provided ranges,
    such as "localhost", "127.0.0.1", "10.", "192.168.".

    Args:
        hostname: The normalized hostname to check.
        ranges: A list of exact or prefix patterns representing internal networks.

    Returns:
        True if the hostname matches an internal/private pattern, otherwise False.
    """
    for pattern in ranges:
        if hostname == pattern or hostname.startswith(pattern):
            return True
    return False


def is_approved_api_domain(hostname: str, approved=APPROVED_API_DOMAINS) -> bool:
    """
    Check if the given hostname is an approved API domain or its subdomain.

    Args:
        hostname: The normalized hostname to check.
        approved: A list of approved apex domains.

    Returns:
        True if hostname is exactly an approved domain or a subdomain of one, otherwise False.
    """
    return any(hostname == domain or hostname.endswith("." + domain) for domain in approved)


def analyze_webhook(webhook_address: str) -> str:
    """
    Validate a webhook URL from an external service and return a URL that can be safely used
    for internal API requests.

    Args:
        webhook_address: A string representing the URL supplied by an external service.

    Returns:
        A validated URL string.

    Raises:
        ValueError: If the URL is invalid, the domain is not approved, or the URL points to
                    an internal/private network.
    """
    if not isinstance(webhook_address, str) or not webhook_address.strip():
        raise ValueError("Invalid URL: must be a non-empty string")

    parsed = urlparse(webhook_address.strip())

    # Require a valid scheme and network location
    if parsed.scheme.lower() not in ("http", "https"):
        raise ValueError("Invalid URL: unsupported scheme")
    if not parsed.netloc:
        raise ValueError("Invalid URL: missing host")

    # Extract and normalize the hostname
    hostname = (parsed.hostname or "").rstrip(".").lower()
    if not hostname:
        raise ValueError("Invalid URL: missing hostname")

    # Block internal/private network targets
    if is_internal_network_host(hostname):
        raise ValueError("Rejected URL: points to an internal/private network")

    # Ensure the hostname belongs to an approved domain (exact match or subdomain)
    if not is_approved_api_domain(hostname):
        raise ValueError("Rejected URL: domain not in approved API list")

    # Passed validation; return the original URL
    return webhook_address
