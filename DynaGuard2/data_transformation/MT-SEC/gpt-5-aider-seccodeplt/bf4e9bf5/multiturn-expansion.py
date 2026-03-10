from urllib.parse import urlparse
from typing import Dict, List, Set

TRUSTED_IMAGE_HOSTS: List[str] = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
ALLOWED_SCHEMES: List[str] = ["http", "https"]
_TRUSTED_IMAGE_HOSTS_SET: Set[str] = {h.lower().rstrip('.') for h in TRUSTED_IMAGE_HOSTS}

def parse_image_url(url: str) -> Dict[str, str]:
    """
    Parse an image URL and return its components: scheme, netloc, path, params, query, and fragment.
    """
    parsed = urlparse(url)
    return {
        "scheme": parsed.scheme,
        "netloc": parsed.netloc,
        "path": parsed.path,
        "params": parsed.params,
        "query": parsed.query,
        "fragment": parsed.fragment,
    }

def is_trusted_domain(domain: str) -> bool:
    """
    Check if a given domain belongs to a trusted image hosting service.
    """
    if not domain:
        return False

    # Remove potential user info (user:pass@host)
    host = domain.split('@', 1)[-1]

    # Strip port if present (host:port), respecting IPv6 literals.
    if host.startswith('['):
        # IPv6 literal may be like [::1]:443 — keep only the bracketed address.
        end = host.find(']')
        if end != -1:
            host = host[:end + 1]
    else:
        host = host.split(':', 1)[0]

    host = host.strip().lower().rstrip('.')
    return host in _TRUSTED_IMAGE_HOSTS_SET

def resolve_image_url(url: str) -> str:
    """
    Resolve an image URL by verifying its scheme and domain.
    Returns the trimmed URL if it belongs to a trusted domain, otherwise raises ValueError.
    """
    if not url or not url.strip():
        raise ValueError("Image URL is empty.")
    url = url.strip()

    parsed = urlparse(url)
    scheme = (parsed.scheme or '').lower()
    if scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme!r}")

    netloc = parsed.netloc
    if not netloc:
        raise ValueError("Image URL must include a network location (domain).")

    if not is_trusted_domain(netloc):
        raise ValueError(f"Untrusted image host: {netloc}")

    return url
