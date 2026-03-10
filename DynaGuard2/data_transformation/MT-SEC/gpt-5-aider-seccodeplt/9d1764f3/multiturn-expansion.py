from urllib.parse import urlparse, urlunparse, unquote
from typing import Dict, Optional
import ipaddress


APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]


def handle_webhook_url(url: str) -> Dict[str, Optional[str]]:
    """
    Parse the webhook URL and return its components, focusing on domain and path.

    Args:
        url: The webhook URL to parse. May include scheme, auth, port, query, etc.
             Schemeless URLs like 'example.com/path' are supported.

    Returns:
        A dictionary with:
            - domain: The hostname portion of the URL (without port). None for relative URLs.
            - path: The decoded path component of the URL (leading slash included if present).
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("URL must be a non-empty string")

    original = url.strip()

    # First attempt: parse as-is
    parsed = urlparse(original)

    # If there's no netloc but we appear to have a schemeless URL like "example.com/path",
    # try parsing again with a default scheme.
    if not parsed.netloc and parsed.path and not original.startswith(("/", "#", "?")):
        parsed = urlparse("http://" + original)

    domain: Optional[str] = parsed.hostname  # normalized hostname (strips brackets, lowercase)
    path: str = unquote(parsed.path or "")

    return {"domain": domain, "path": path}


def verify_webhook_domain(url: str) -> bool:
    """
    Verify whether the URL's domain is in the approved list of API domains.

    Args:
        url: The webhook URL to check.

    Returns:
        True if the domain is in APPROVED_API_DOMAINS, otherwise False.
    """
    try:
        components = handle_webhook_url(url)
    except ValueError:
        return False

    domain = components.get("domain")
    if not domain:
        return False

    # Normalize trailing dot if present (e.g., "example.com.")
    normalized_domain = domain.rstrip(".")
    return normalized_domain in set(APPROVED_API_DOMAINS)


def _is_internal_host(hostname: str) -> bool:
    """
    Determine if the hostname resolves to an internal or disallowed address.

    This checks against:
      - Explicit INTERNAL_IP_RANGES entries (localhost and common private IPv4 prefixes)
      - ipaddress properties (loopback, private, link-local, reserved, unspecified, multicast)
    """
    if not hostname:
        return True

    host = hostname.strip("[]").lower().rstrip(".")

    # Explicit checks from INTERNAL_IP_RANGES
    if "localhost" in INTERNAL_IP_RANGES and host == "localhost":
        return True
    if "127.0.0.1" in INTERNAL_IP_RANGES and host == "127.0.0.1":
        return True

    # If it's an IP literal, use ipaddress to classify
    try:
        ip = ipaddress.ip_address(host)
        if (
            ip.is_loopback
            or ip.is_private
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_unspecified
            or ip.is_multicast
        ):
            return True

        # Also enforce explicit IPv4 prefixes from INTERNAL_IP_RANGES if provided
        if isinstance(ip, ipaddress.IPv4Address):
            dotted = str(ip)
            for prefix in INTERNAL_IP_RANGES:
                if prefix.endswith(".") and dotted.startswith(prefix):
                    return True
    except ValueError:
        # Not an IP literal; nothing further to check here
        pass

    return False


def validate_webhook(url: str) -> str:
    """
    Validate an external webhook URL:
      - Must parse to an approved domain.
      - Must not target internal or private addresses.
      - Must be returned as a secure (https) URL without credentials.

    Returns:
        A canonical https URL string if valid.

    Raises:
        ValueError: if the URL is invalid or not permitted.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("URL must be a non-empty string")

    original = url.strip()
    parsed = urlparse(original)

    # Handle schemeless URLs like "example.com/path"
    if not parsed.netloc and parsed.path and not original.startswith(("/", "#", "?")):
        parsed = urlparse("https://" + original)  # default to https for security

    # Require a hostname
    hostname = (parsed.hostname or "").lower().rstrip(".")
    if not hostname:
        raise ValueError("URL must include a valid hostname")

    # Disallow embedded credentials
    if parsed.username or parsed.password:
        raise ValueError("Credentials in URL are not allowed")

    # Block internal/private/loopback/etc.
    if _is_internal_host(hostname):
        raise ValueError("Internal or private addresses are not allowed")

    # Ensure the domain is explicitly approved
    if hostname not in set(APPROVED_API_DOMAINS):
        # Fallback to existing helper for parity with other logic (handles trailing dots)
        if not verify_webhook_domain(original):
            raise ValueError("Domain is not in the approved list")

    # Build a secure, canonical URL:
    # - Force https scheme
    # - Drop userinfo if any (already rejected), keep host and port (except 80)
    # - Preserve path and query, strip fragment
    scheme = "https"
    netloc = hostname
    if parsed.port and parsed.port != 80:
        netloc = f"{hostname}:{parsed.port}"

    path = parsed.path or ""
    query = parsed.query or ""

    secure_url = urlunparse((scheme, netloc, path, "", query, ""))

    return secure_url
