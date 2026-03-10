from urllib.parse import urlparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]


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

    # Block internal/private network targets based on simple prefix checks
    for rng in INTERNAL_IP_RANGES:
        # Exact match or prefix match for ranges like "10." and "192.168."
        if hostname == rng or hostname.startswith(rng):
            raise ValueError("Rejected URL: points to an internal/private network")

    # Ensure the hostname belongs to an approved domain (exact match or subdomain)
    is_allowed = any(
        hostname == allowed or hostname.endswith("." + allowed)
        for allowed in APPROVED_API_DOMAINS
    )
    if not is_allowed:
        raise ValueError("Rejected URL: domain not in approved API list")

    # Passed validation; return the original URL
    return webhook_address
