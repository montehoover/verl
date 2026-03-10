from urllib.parse import urlparse, unquote
from typing import Dict, Optional


APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]


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
