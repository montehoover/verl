from urllib.parse import urlparse, urlunparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]


def is_internal_host(host: str, internal_ranges=INTERNAL_IP_RANGES) -> bool:
    """
    Determine if a hostname points to an internal or loopback location.
    Returns True if the host is internal, False otherwise.
    """
    host_lc = (host or "").lower()
    for rng in internal_ranges:
        rng_lc = rng.lower()
        if rng_lc.endswith("."):
            if host_lc.startswith(rng_lc):
                return True
        else:
            if host_lc == rng_lc:
                return True
    return False


def is_approved_domain(host: str, approved_domains=APPROVED_API_DOMAINS) -> bool:
    """
    Check if the hostname exactly matches one of the approved API domains.
    """
    approved = {d.lower() for d in approved_domains}
    return (host or "").lower() in approved


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

    if is_internal_host(host):
        raise ValueError("URL targets an internal network location")

    if not is_approved_domain(host):
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
