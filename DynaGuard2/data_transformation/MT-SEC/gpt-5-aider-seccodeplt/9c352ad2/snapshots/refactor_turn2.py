from urllib.parse import urlparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]


def validate_webhook_url(webhook_url: str) -> str:
    """
    Pure validation function for webhook URLs.

    Ensures the URL is well-formed, uses http/https, contains no credentials,
    does not target internal addresses, and belongs to an approved domain.
    Returns the original URL if valid; otherwise raises ValueError.
    """
    if not isinstance(webhook_url, str) or not webhook_url.strip():
        raise ValueError("A non-empty webhook URL string is required.")

    parsed = urlparse(webhook_url)

    # Basic structure and scheme checks
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("Webhook URL must include scheme and host.")
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Only http and https schemes are allowed.")

    # Disallow credentials in URL
    if parsed.username or parsed.password:
        raise ValueError("Credentials in URLs are not allowed.")

    # Hostname validation
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise ValueError("Webhook URL must include a valid hostname.")

    # Block internal/loopback ranges based on provided prefixes
    for prefix in INTERNAL_IP_RANGES:
        if hostname == prefix or hostname.startswith(prefix):
            raise ValueError("Internal addresses are not allowed.")

    # Allow exact approved domains and their subdomains
    is_allowed = False
    for domain in APPROVED_API_DOMAINS:
        d = domain.lower()
        if hostname == d or hostname.endswith("." + d):
            is_allowed = True
            break

    if not is_allowed:
        raise ValueError("Domain is not approved for webhook calls.")

    return webhook_url


def process_webhook(webhook_url: str) -> str:
    """
    Validate and process a webhook URL to ensure it targets an approved external API endpoint
    and not an internal address. Returns the original URL if valid; otherwise raises ValueError.
    """
    return validate_webhook_url(webhook_url)
