from urllib.parse import urlparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]


def handle_webhook_request(external_url: str) -> str:
    """
    Validate and sanitize a webhook URL provided by an external service.

    Args:
        external_url: A string representing the URL supplied by an external service.

    Returns:
        A validated URL that can be safely used for internal API requests.

    Raises:
        ValueError: If the URL is invalid, the domain is not approved, or the URL
                    points to an internal/private network.
    """
    # Basic type and emptiness check
    if not isinstance(external_url, str) or not external_url.strip():
        raise ValueError("external_url must be a non-empty string")

    # Parse the URL
    parsed = urlparse(external_url.strip())

    # Validate scheme and hostname
    if parsed.scheme not in ("http", "https"):
        raise ValueError("URL scheme must be http or https")

    host = parsed.hostname
    if not host:
        raise ValueError("URL must include a valid host")

    host_lc = host.lower()

    # Block internal/private network targets based on provided prefixes
    for private in INTERNAL_IP_RANGES:
        pl = private.lower()
        if host_lc == pl or host_lc.startswith(pl):
            raise ValueError("URL points to an internal/private network")

    # Only allow approved public API domains (exact host match)
    approved_hosts_lc = [d.lower() for d in APPROVED_API_DOMAINS]
    if host_lc not in approved_hosts_lc:
        raise ValueError("Domain is not in the approved API domain list")

    # Return the original (trimmed) URL if all checks pass
    return external_url.strip()
