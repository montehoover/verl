from urllib.parse import urlparse

ALLOWED_REDIRECT_DOMAINS = ["trusted-redirect.com", "partner.com"]

def extract_url_components(url: str):
    """
    Parse the given URL and return its components.

    Returns a dictionary with keys: scheme, netloc, path, params, query, fragment.
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

def _normalize_host(host: str | None) -> str | None:
    """
    Normalize a hostname for comparison:
    - lower-case
    - strip any trailing dot
    """
    if not host:
        return None
    return host.rstrip(".").lower()

def is_domain_allowed(url: str) -> bool:
    """
    Check whether the URL's domain is in the allowed domains list.
    Matching is exact against the hostname (no subdomain wildcards).
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return False

    host = _normalize_host(parsed.hostname)
    if not host:
        return False

    allowed = {d.rstrip(".").lower() for d in ALLOWED_REDIRECT_DOMAINS}
    return host in allowed
