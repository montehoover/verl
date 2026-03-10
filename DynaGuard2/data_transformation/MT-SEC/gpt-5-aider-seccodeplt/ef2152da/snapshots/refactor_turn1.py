from urllib.parse import urlparse, urlunparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]


def validate_callback(callback_url: str) -> str:
    """
    Validate and sanitize a callback URL for internal API use.

    - Ensures the URL uses http/https scheme (and enforces https for security).
    - Ensures the hostname is exactly one of the approved API domains.
    - Blocks internal/private network targets based on INTERNAL_IP_RANGES.
    - Rejects URLs with embedded credentials (userinfo).
    - Returns a normalized, secure (https) URL suitable for internal calls.

    Raises:
        ValueError: If the URL is invalid, uses disallowed domains, targets
                    internal/private networks, or contains userinfo.
    """
    if not isinstance(callback_url, str) or not callback_url.strip():
        raise ValueError("callback_url must be a non-empty string URL")

    parsed = urlparse(callback_url.strip())

    if parsed.scheme not in ("http", "https"):
        raise ValueError("URL scheme must be http or https")

    # Reject userinfo to avoid credential leakage in SSRF scenarios
    # ParseResult exposes username/password when present (derived from netloc).
    if parsed.username is not None or parsed.password is not None or ("@" in parsed.netloc):
        raise ValueError("URL must not contain user information")

    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise ValueError("URL must include a valid hostname")

    # Block internal/private network targets
    if hostname == "localhost" or hostname == "127.0.0.1":
        raise ValueError("Access to internal/private network addresses is not allowed")
    if hostname.startswith("10.") or hostname.startswith("192.168."):
        raise ValueError("Access to internal/private network addresses is not allowed")

    # Enforce whitelist of approved public API endpoints (exact match)
    if hostname not in APPROVED_API_DOMAINS:
        raise ValueError("URL domain is not in the approved API endpoint list")

    # Enforce HTTPS for internal API interactions
    secure_scheme = "https"

    # Reconstruct a normalized URL (drop fragments)
    netloc = hostname
    if parsed.port:
        netloc = f"{hostname}:{parsed.port}"

    sanitized = urlunparse(
        (
            secure_scheme,
            netloc,
            parsed.path or "",
            parsed.params or "",
            parsed.query or "",
            "",  # fragment removed
        )
    )

    return sanitized
