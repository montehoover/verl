from urllib.parse import urlparse, urlunparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]


def handle_webhook_request(external_url: str) -> str:
    """
    Process and validate a webhook URL supplied by an external service.

    Args:
        external_url: The URL provided by an external service.

    Returns:
        A validated and sanitized URL safe for internal API requests.

    Raises:
        ValueError: If the URL is invalid, contains credentials, the domain is not
                    in the approved list, or the URL points to an internal/private network.
    """
    if not isinstance(external_url, str) or not external_url.strip():
        raise ValueError("Invalid URL: empty or non-string value provided")

    raw = external_url.strip()
    parsed = urlparse(raw)

    # Require explicit scheme and host
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Invalid or unsupported URL scheme")
    if not parsed.netloc:
        raise ValueError("Invalid URL: missing host")

    # Disallow embedded credentials
    if parsed.username or parsed.password:
        raise ValueError("URL must not contain user credentials")

    hostname = (parsed.hostname or "").lower().rstrip(".")
    if not hostname:
        raise ValueError("Invalid URL: missing hostname")

    # Block internal/private networks using configured prefixes
    if (
        hostname == "localhost"
        or hostname.startswith("127.")
        or hostname.startswith("10.")
        or hostname.startswith("192.168.")
    ):
        raise ValueError("URL points to an internal/private network and is not allowed")

    # Ensure the hostname is exactly one of the approved public API domains
    if hostname not in {d.lower() for d in APPROVED_API_DOMAINS}:
        raise ValueError("Domain is not in the approved API endpoint list")

    # Reconstruct a sanitized URL (strip credentials, preserve port/path/query)
    netloc = hostname
    if parsed.port:
        netloc = f"{hostname}:{parsed.port}"

    sanitized = urlunparse(
        (parsed.scheme, netloc, parsed.path or "/", parsed.params, parsed.query, "")
    )
    return sanitized
