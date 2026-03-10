from urllib.parse import urlparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]


def process_webhook(webhook_url: str) -> str:
    """
    Validate and return a webhook URL that points to a pre-approved API endpoint.

    Args:
        webhook_url: The URL provided by an external service.

    Returns:
        The validated URL string.

    Raises:
        ValueError: If the URL is empty/invalid, uses a non-http(s) scheme,
                    resolves to an internal address, or the domain is not approved.
    """
    if not isinstance(webhook_url, str):
        raise ValueError("webhook_url must be a string")

    url = webhook_url.strip()
    if not url:
        raise ValueError("webhook_url must not be empty")

    parsed = urlparse(url)

    # Require an explicit http(s) scheme
    if parsed.scheme not in ("http", "https"):
        raise ValueError("URL scheme must be http or https")

    # hostname omits any username, password, or port; lowercased by urllib
    host = parsed.hostname
    if not host:
        raise ValueError("URL must include a hostname")

    host = host.lower()

    # Block internal addresses by exact or prefix match based on provided ranges
    for prefix in INTERNAL_IP_RANGES:
        if prefix in ("localhost", "127.0.0.1"):
            if host == prefix:
                raise ValueError("Internal addresses are not allowed")
        else:
            if host.startswith(prefix):
                raise ValueError("Internal addresses are not allowed")

    # Only allow exact matches to approved API domains
    if host not in APPROVED_API_DOMAINS:
        raise ValueError("Domain is not an approved API endpoint")

    # Return the original, trimmed URL after validation
    return url
