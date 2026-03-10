from urllib.parse import urlparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]


def process_webhook(webhook_url: str) -> str:
    """
    Validate and process a webhook URL to ensure it targets an approved external API endpoint
    and not an internal address. Returns the original URL if valid; otherwise raises ValueError.

    Args:
        webhook_url: The URL provided by an external service.

    Returns:
        The validated URL string.

    Raises:
        ValueError: If the URL is malformed, uses an unsupported scheme, includes credentials,
                    targets an internal address, or is not within approved domains.
    """
    if not isinstance(webhook_url, str) or not webhook_url.strip():
        raise ValueError("A non-empty webhook URL string is required.")

    parsed = urlparse(webhook_url)

    if not parsed.scheme or not parsed.netloc:
        raise ValueError("Webhook URL must include scheme and host.")
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Only http and https schemes are allowed.")
    # Disallow embedding credentials in the URL to prevent credential leakage.
    if parsed.username or parsed.password:
        raise ValueError("Credentials in URLs are not allowed.")

    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise ValueError("Webhook URL must include a valid hostname.")

    # Block internal/loopback ranges based on provided prefixes.
    for prefix in INTERNAL_IP_RANGES:
        if hostname == prefix or hostname.startswith(prefix):
            raise ValueError("Internal addresses are not allowed.")

    # Allow exact approved domains and their subdomains (with dot-boundary).
    is_allowed = False
    for domain in APPROVED_API_DOMAINS:
        d = domain.lower()
        if hostname == d or hostname.endswith("." + d):
            is_allowed = True
            break

    if not is_allowed:
        raise ValueError("Domain is not approved for webhook calls.")

    return webhook_url
