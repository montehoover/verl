from urllib.parse import urlparse, urlunparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]


def validate_webhook(webhook_link: str) -> str:
    """
    Validate an external webhook URL and return a secure, internal-allowed URL.

    Args:
        webhook_link: External webhook URL string.

    Returns:
        A validated HTTPS URL string suitable for internal API interactions.

    Raises:
        ValueError: If the URL is invalid, uses an unsupported scheme,
                    targets internal/private network addresses, or the domain
                    is not in the approved API endpoint list.
    """
    if not isinstance(webhook_link, str) or not webhook_link.strip():
        raise ValueError("Invalid URL input.")

    parsed = urlparse(webhook_link.strip())

    # Must have scheme and netloc
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("Invalid URL format.")

    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError("Unsupported URL scheme.")

    host = (parsed.hostname or "").lower()
    if not host:
        raise ValueError("Invalid URL host.")

    # Block internal/private network addresses by prefix check
    for prefix in INTERNAL_IP_RANGES:
        if host == prefix or host.startswith(prefix):
            raise ValueError("URL points to a disallowed internal/private address.")

    # Enforce allowlist of API domains (exact match)
    if host not in APPROVED_API_DOMAINS:
        raise ValueError("URL domain is not approved.")

    # Force HTTPS for secure internal interactions
    secure_scheme = "https"

    # Reconstruct sanitized netloc without userinfo
    netloc = host
    if parsed.port:
        netloc = f"{host}:{parsed.port}"

    path = parsed.path or "/"
    query = parsed.query or ""

    secure_url = urlunparse((secure_scheme, netloc, path, "", query, ""))

    return secure_url
