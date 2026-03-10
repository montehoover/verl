from urllib.parse import urlparse
from typing import Dict, List, Optional

ALLOWED_REDIRECT_DOMAINS = ["trusted-redirect.com", "partner.com"]


def extract_url_components(url: str) -> Dict[str, str]:
    """
    Parse a URL and return its components.

    Returns a dict with keys: scheme, netloc, path, params, query, fragment.
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


def is_domain_allowed(url: str) -> bool:
    """
    Check if the URL's domain is in the allowed redirect domains list.
    Matches exact domains and their subdomains.
    """
    parsed = urlparse(url)

    # Handle URLs without scheme like "example.com/path"
    if not parsed.netloc and "://" not in url and not url.startswith("//"):
        parsed = urlparse(f"http://{url}")

    host = parsed.hostname
    if not host:
        return False

    host = host.rstrip(".").lower()
    try:
        host_idna = host.encode("idna").decode("ascii")
    except Exception:
        host_idna = host

    for allowed in ALLOWED_REDIRECT_DOMAINS:
        a = allowed.rstrip(".").lower()
        try:
            a_idna = a.encode("idna").decode("ascii")
        except Exception:
            a_idna = a

        if host_idna == a_idna or host_idna.endswith("." + a_idna):
            return True

    return False
