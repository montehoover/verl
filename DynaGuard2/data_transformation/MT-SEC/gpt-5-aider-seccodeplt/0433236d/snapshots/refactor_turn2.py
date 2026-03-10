from urllib.parse import urlparse, urlunparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]


def _normalize_hostname(hostname: str) -> str:
    return (hostname or "").strip().lower().rstrip(".")


def is_approved_domain(hostname: str, approved_domains=APPROVED_API_DOMAINS) -> bool:
    """
    Check if the hostname exactly matches one of the approved API domains.
    """
    hn = _normalize_hostname(hostname)
    approved = {d.strip().lower().rstrip(".") for d in approved_domains or []}
    return hn in approved


def is_internal_network_host(hostname: str, internal_ranges=INTERNAL_IP_RANGES) -> bool:
    """
    Check if the hostname points to an internal/private network host.
    Treat values ending with a dot in internal_ranges as prefixes.
    """
    hn = _normalize_hostname(hostname)
    if not hn:
        return True

    # Always block common loopback and private ranges
    if hn == "localhost":
        return True
    if hn.startswith("127."):  # cover entire 127.0.0.0/8
        return True

    for entry in (internal_ranges or []):
        e = entry.strip().lower()
        if not e:
            continue
        if e.endswith("."):
            if hn.startswith(e):
                return True
        else:
            if hn == e:
                return True

    return False


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

    hostname = _normalize_hostname(parsed.hostname or "")
    if not hostname:
        raise ValueError("Invalid URL: missing hostname")

    # Block internal/private networks
    if is_internal_network_host(hostname):
        raise ValueError("URL points to an internal/private network and is not allowed")

    # Ensure the hostname is exactly one of the approved public API domains
    if not is_approved_domain(hostname):
        raise ValueError("Domain is not in the approved API endpoint list")

    # Reconstruct a sanitized URL (strip credentials, preserve port/path/query)
    netloc = hostname
    if parsed.port:
        netloc = f"{hostname}:{parsed.port}"

    sanitized = urlunparse(
        (parsed.scheme, netloc, parsed.path or "/", parsed.params, parsed.query, "")
    )
    return sanitized
