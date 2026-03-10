from urllib.parse import urlparse
from typing import Dict, List, Set

TRUSTED_IMAGE_HOSTS: List[str] = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
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
