from urllib.parse import urlparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]

_APPROVED_API_DOMAINS_LOWER = {d.lower() for d in APPROVED_API_DOMAINS}


def analyze_webhook(webhook_address: str) -> str:
    """
    Validate and normalize an external webhook URL for internal API usage.

    Args:
        webhook_address: A string representing the URL supplied by an external service.

    Returns:
        The validated URL string that can be safely used for internal API requests.

    Raises:
        ValueError: If the URL is invalid, the domain is not approved, or it points
                    to an internal/private network.
    """
    if not isinstance(webhook_address, str):
        raise ValueError("webhook_address must be a string")

    url = webhook_address.strip()
    if not url:
        raise ValueError("Empty URL")

    parsed = urlparse(url)

    # Basic URL structure checks
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Unsupported or missing URL scheme; only http and https are allowed")
    if not parsed.netloc:
        raise ValueError("URL must include a network location (hostname)")

    # Hostname extracted in a normalized (lowercased) form, without port/userinfo.
    host = (parsed.hostname or "").lower()
    if not host:
        raise ValueError("URL must include a valid hostname")

    # Reject internal/private network targets based on simple prefix checks
    # (as provided by INTERNAL_IP_RANGES).
    for prefix in INTERNAL_IP_RANGES:
        if host.startswith(prefix):
            raise ValueError("URL points to an internal/private network and is not allowed")

    # Ensure the host is exactly one of the approved API domains.
    if host not in _APPROVED_API_DOMAINS_LOWER:
        raise ValueError("Domain is not in the list of approved API endpoints")

    # If all checks pass, return the original (trimmed) URL.
    return url
