from urllib.parse import urlparse, urlunparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]


def process_hook(hook_url: str) -> str:
    """
    Validate a webhook URL and return a safely validated URL that references an approved API endpoint.

    Raises:
        ValueError: If the URL is invalid, uses a non-http(s) scheme, points to an internal location,
                    or the domain is not in the approved list.
    """
    if not isinstance(hook_url, str):
        raise ValueError("hook_url must be a string")

    candidate = hook_url.strip()
    if not candidate:
        raise ValueError("hook_url must not be empty")

    parsed = urlparse(candidate)

    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        raise ValueError("URL scheme must be http or https")

    host = (parsed.hostname or "").lower()
    if not host:
        raise ValueError("URL must include a valid hostname")

    # Block internal/localhost targets
    for rng in INTERNAL_IP_RANGES:
        if rng.endswith("."):
            if host.startswith(rng.lower()):
                raise ValueError("URL targets an internal network location")
        else:
            if host == rng.lower():
                raise ValueError("URL targets an internal network location")

    # Ensure hostname is in the approved list (exact match)
    approved = {d.lower() for d in APPROVED_API_DOMAINS}
    if host not in approved:
        raise ValueError("URL domain is not an approved API endpoint")

    # Return a sanitized form (no fragment)
    sanitized = urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path or "",
            parsed.params,
            parsed.query,
            "",
        )
    )
    return sanitized
