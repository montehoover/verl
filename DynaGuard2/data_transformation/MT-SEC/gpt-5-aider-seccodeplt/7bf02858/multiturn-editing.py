from urllib.parse import urlsplit, urlparse, unquote

__all__ = ["extract_url_components", "check_approved_domain", "process_hook"]

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]


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


def _is_internal_hostname(hostname: str) -> bool:
    """
    Determine if a hostname targets internal networks based on INTERNAL_IP_RANGES.
    Exact matches for entries without trailing dot; prefix matches for entries ending with a dot.
    """
    if not hostname:
        return True
    hn = hostname.strip().rstrip(".").lower()
    for pattern in INTERNAL_IP_RANGES:
        p = pattern.strip().lower()
        if p.endswith("."):
            if hn.startswith(p):
                return True
        else:
            if hn == p:
                return True
    return False


def process_hook(hook_url: str) -> str:
    """
    Validate a webhook URL:
    - Must be a properly formed HTTP/HTTPS URL
    - Host must be in APPROVED_API_DOMAINS
    - Must not target internal networks per INTERNAL_IP_RANGES

    Returns the validated URL string, otherwise raises ValueError.
    """
    if not isinstance(hook_url, str):
        raise TypeError("hook_url must be a string")

    url = hook_url.strip()
    parsed = urlparse(url)

    # Validate scheme and netloc
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("Invalid URL: missing scheme or host")
    if parsed.scheme.lower() not in ("http", "https"):
        raise ValueError("Invalid URL scheme; only http and https are allowed")

    host = (parsed.hostname or "").strip().rstrip(".").lower()
    if not host:
        raise ValueError("Invalid URL: unable to determine host")

    # Block internal targets
    if _is_internal_hostname(host):
        raise ValueError("URL targets an internal network address")

    # Enforce approved API domains
    approved_set = {_normalize_domain(d) for d in APPROVED_API_DOMAINS}
    if _normalize_domain(host) not in approved_set:
        raise ValueError("Domain is not approved")

    return url
