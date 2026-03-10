from urllib.parse import urlparse, urlunparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]


def is_internal_or_private_host(host: str, internal_ranges: list[str]) -> bool:
    """
    Determine if the given host points to an internal or private network address.
    Checks for exact matches and prefix-based ranges as provided.

    Args:
        host: The hostname (lowercased) extracted from the URL.
        internal_ranges: A list containing exact hostnames or prefixes that identify
                         internal/private addresses.

    Returns:
        True if the host should be considered internal/private; otherwise False.
    """
    if not host:
        return True  # Treat empty host as invalid/internal
    host_lc = host.lower()
    for prefix in internal_ranges:
        if host_lc == prefix or host_lc.startswith(prefix):
            return True
    return False


def is_approved_domain(host: str, approved_domains: list[str]) -> bool:
    """
    Check if the given host exactly matches one of the approved API domains.

    Args:
        host: The hostname (lowercased) extracted from the URL.
        approved_domains: The allowlist of approved domains.

    Returns:
        True if host is an exact match for an approved domain; otherwise False.
    """
    if not host:
        return False
    host_lc = host.lower()
    return any(host_lc == d.lower() for d in approved_domains)


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
    if is_internal_or_private_host(host, INTERNAL_IP_RANGES):
        raise ValueError("URL points to a disallowed internal/private address.")

    # Enforce allowlist of API domains (exact match)
    if not is_approved_domain(host, APPROVED_API_DOMAINS):
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
