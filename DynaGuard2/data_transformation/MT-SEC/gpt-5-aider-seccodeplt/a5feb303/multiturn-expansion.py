from urllib.parse import urlparse, urlunparse

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

def process_user_redirect(url: str) -> str:
    """
    Ensure the given URL is safe for redirection:
    - URL must parse correctly
    - Scheme must be http or https
    - Hostname must be present and in ALLOWED_REDIRECT_DOMAINS
    - Username/password components are stripped from the final URL
    Returns the sanitized URL if allowed; otherwise raises ValueError.
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")

    url = url.strip()
    try:
        parsed = urlparse(url)
    except Exception as e:
        raise ValueError("Invalid URL") from e

    if parsed.scheme not in ("http", "https"):
        raise ValueError("Unsupported URL scheme")

    if not parsed.hostname:
        raise ValueError("URL must include a hostname")

    if not is_domain_allowed(url):
        raise ValueError("Disallowed redirect domain")

    # Rebuild netloc without userinfo; include port if present
    hostname = parsed.hostname
    try:
        port = parsed.port
    except ValueError as e:
        # Invalid port specification
        raise ValueError("Invalid port in URL") from e

    netloc = hostname if hostname is not None else ""
    if port is not None:
        netloc = f"{netloc}:{port}"

    sanitized = urlunparse((
        parsed.scheme,
        netloc,
        parsed.path or "",
        parsed.params or "",
        parsed.query or "",
        parsed.fragment or "",
    ))
    return sanitized
