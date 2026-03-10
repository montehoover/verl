from urllib.parse import urlsplit, unquote

__all__ = ["extract_url_components", "check_approved_domain"]


def extract_url_components(url: str) -> dict:
    """
    Extract components from a URL string.

    Returns a dict with keys:
    - scheme
    - domain
    - port
    - path
    - query
    - fragment
    - username
    - password
    """
    if not isinstance(url, str):
        raise TypeError("url must be a string")

    s = url.strip()

    # Allow schemeless host URLs like "example.com/path" by prefixing '//'
    if "://" not in s and not s.startswith("//") and not s.startswith("/"):
        s_for_parse = "//" + s
    else:
        s_for_parse = s

    parsed = urlsplit(s_for_parse, allow_fragments=True)

    path = unquote(parsed.path or "")
    if path == "" and parsed.netloc:
        path = "/"

    return {
        "scheme": parsed.scheme or None,
        "domain": parsed.hostname,
        "port": parsed.port,
        "path": path,
        "query": parsed.query or "",
        "fragment": parsed.fragment or "",
        "username": parsed.username,
        "password": parsed.password,
    }


def _normalize_domain(domain: str) -> str:
    """
    Normalize a domain for comparison:
    - lowercase
    - strip surrounding whitespace
    - remove any trailing dot
    """
    return domain.strip().rstrip(".").lower()


def check_approved_domain(url: str, approved_domains) -> dict:
    """
    Check whether the URL's domain is within the approved domains.

    Parameters:
    - url: URL string to check
    - approved_domains: an iterable (list/tuple/set) of domain strings

    Returns:
    {
        "approved": bool,        # True if domain is approved, else False
        "message": "approved" | "not approved"
    }
    """
    if not isinstance(url, str):
        raise TypeError("url must be a string")

    if not isinstance(approved_domains, (list, tuple, set)):
        raise TypeError("approved_domains must be a list, tuple, or set of domain strings")

    components = extract_url_components(url)
    domain = components.get("domain")

    if not domain:
        return {"approved": False, "message": "not approved"}

    approved_set = {_normalize_domain(d) for d in approved_domains if isinstance(d, str)}
    is_approved = _normalize_domain(domain) in approved_set

    return {"approved": is_approved, "message": "approved" if is_approved else "not approved"}
