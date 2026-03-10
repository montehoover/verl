from urllib.parse import urlparse, urlunparse
import ipaddress

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]


def validate_callback(callback_url: str) -> str:
    """
    Validate a webhook callback URL and return a sanitized, secure URL that can be used
    for internal API interactions.

    Rules:
    - Must use HTTPS.
    - Hostname must be exactly in APPROVED_API_DOMAINS.
    - Must not target internal/private/loopback addresses (e.g., localhost, 127.0.0.1, 10.0.0.0/8, 192.168.0.0/16).
    - Returns a normalized URL without userinfo or fragments.

    Raises:
        ValueError: If the URL is invalid, not approved, or targets internal/private addresses.
    """
    if not isinstance(callback_url, str) or not callback_url.strip():
        raise ValueError("callback_url must be a non-empty string.")

    parsed = urlparse(callback_url.strip())

    # Require HTTPS for secure internal interactions
    scheme = (parsed.scheme or "").lower()
    if scheme != "https":
        raise ValueError("Only HTTPS callback URLs are allowed.")

    host = parsed.hostname  # hostname excludes userinfo and brackets
    if not host:
        raise ValueError("Callback URL must include a valid hostname.")

    host_l = host.lower()

    # Block obvious internal targets by prefix/keyword
    for prefix in INTERNAL_IP_RANGES:
        if host_l.startswith(prefix):
            raise ValueError("Callback URL targets a disallowed internal/private address.")

    # If the host is an IP literal, use ipaddress to detect private/loopback/etc.
    try:
        ip_obj = ipaddress.ip_address(host_l)
    except ValueError:
        ip_obj = None

    if ip_obj:
        if (
            ip_obj.is_private
            or ip_obj.is_loopback
            or ip_obj.is_link_local
            or ip_obj.is_reserved
            or ip_obj.is_multicast
        ):
            raise ValueError("Callback URL targets a disallowed internal/private IP address.")

    # Enforce exact match to approved API domains
    if host_l not in (d.lower() for d in APPROVED_API_DOMAINS):
        raise ValueError("Callback URL domain is not in the approved API endpoint list.")

    # Rebuild a sanitized URL: no userinfo, keep port if present, strip fragments
    netloc = host_l
    if parsed.port:
        netloc = f"{host_l}:{parsed.port}"

    path = parsed.path or "/"
    sanitized = urlunparse((scheme, netloc, path, parsed.params, parsed.query, ""))

    return sanitized
