from urllib.parse import urlparse, urlunparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]


def process_hook(hook_url: str) -> str:
    """
    Validate and sanitize a webhook URL from an external service.

    Args:
        hook_url: A string representing the URL provided by the external service.

    Returns:
        A safely validated URL that references an approved API endpoint.

    Raises:
        ValueError: If the URL is malformed, uses an unsupported scheme, the domain
                    is outside the list of approved API endpoints, or the URL targets
                    an internal network location.
    """
    if not isinstance(hook_url, str):
        raise ValueError("hook_url must be a string")

    url = hook_url.strip()
    if not url:
        raise ValueError("hook_url must not be empty")

    parsed = urlparse(url)

    # Require a standard web scheme and a host
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("Invalid URL format")
    if parsed.scheme.lower() not in ("http", "https"):
        raise ValueError("Unsupported URL scheme")

    # Extract hostname (lowercased, without userinfo/port/brackets)
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL must include a valid hostname")
    host = hostname.lower()

    # Block internal network targets by simple hostname/prefix checks
    for internal in INTERNAL_IP_RANGES:
        if host == internal or host.startswith(internal):
            raise ValueError("URL points to an internal network location")

    # Enforce exact match against approved public API domains
    approved_lower = {d.lower() for d in APPROVED_API_DOMAINS}
    if host not in approved_lower:
        raise ValueError("URL domain is not approved")

    # Rebuild a sanitized URL:
    # - remove any userinfo
    # - keep scheme, hostname, optional port, path, and query
    netloc = host
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"

    sanitized = urlunparse((
        parsed.scheme.lower(),
        netloc,
        parsed.path or "",
        "",  # params (deprecated in modern URLs)
        parsed.query or "",
        ""   # fragment (not sent to server)
    ))

    return sanitized
