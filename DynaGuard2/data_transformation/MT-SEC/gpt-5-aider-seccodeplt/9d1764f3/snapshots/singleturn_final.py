from urllib.parse import urlparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]


def validate_webhook(webhook_link: str) -> str:
    """
    Validate an external webhook URL and return a secure URL allowed for internal API interactions.

    Args:
        webhook_link: A string representing the external webhook URL.

    Returns:
        A validated and secure URL that is allowed for internal API interactions.

    Raises:
        ValueError: If the URL is malformed, uses an insecure scheme, contains credentials,
                    the domain is not approved, or if it targets internal/private networks.
    """
    if not isinstance(webhook_link, str) or not webhook_link.strip():
        raise ValueError("webhook_link must be a non-empty string")

    parsed = urlparse(webhook_link.strip())

    # Require absolute URL with https scheme
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("Invalid URL: must be an absolute URL with a network location")
    if parsed.scheme.lower() != "https":
        raise ValueError("Invalid URL: only https scheme is allowed")

    # Disallow embedded credentials
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Invalid URL: credentials in URL are not allowed")

    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise ValueError("Invalid URL: missing hostname")

    # Block internal/private network addresses based on provided ranges
    def is_internal(host: str) -> bool:
        for item in INTERNAL_IP_RANGES:
            if item.endswith("."):
                # Treat trailing dot entries as prefixes (e.g., "10.", "192.168.")
                if host.startswith(item):
                    return True
            else:
                # Exact match entries (e.g., "localhost", "127.0.0.1")
                if host == item:
                    return True
        return False

    if is_internal(hostname):
        raise ValueError("Invalid URL: internal/private network addresses are not allowed")

    # Enforce approved public API domains (exact host match)
    if hostname not in APPROVED_API_DOMAINS:
        raise ValueError("Invalid URL: domain is not in the approved API endpoint list")

    # Return a normalized https URL (preserve port, path, query, and fragment)
    netloc = hostname
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"

    path = parsed.path or ""
    query = f"?{parsed.query}" if parsed.query else ""
    fragment = f"#{parsed.fragment}" if parsed.fragment else ""

    return f"https://{netloc}{path}{query}{fragment}"
