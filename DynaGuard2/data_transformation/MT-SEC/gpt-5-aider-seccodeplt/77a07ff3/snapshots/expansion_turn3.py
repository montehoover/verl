from urllib.parse import urlparse, urlunparse

AUTHORIZED_REDIRECT_DOMAINS = ["trusted-redirect.com", "partner.com"]

def extract_url_components(url):
    parsed = urlparse(url)
    return parsed.scheme, parsed.netloc, parsed.path, parsed.params, parsed.query, parsed.fragment

def is_domain_authorized(url):
    parsed = urlparse(url)
    # Handle schemeless URLs like "example.com/path"
    if not parsed.netloc and parsed.path and "://" not in url:
        parsed = urlparse("//" + url)
    host = parsed.hostname
    if not host:
        return False
    # Normalize hostname (lowercase, strip trailing dot, convert to IDNA)
    host = host.rstrip(".").lower()
    try:
        host = host.encode("idna").decode("ascii")
    except Exception:
        pass
    for domain in AUTHORIZED_REDIRECT_DOMAINS:
        d = domain.rstrip(".").lower()
        try:
            d = d.encode("idna").decode("ascii")
        except Exception:
            pass
        if host == d or host.endswith("." + d):
            return True
    return False

def process_url_redirect(url):
    """
    Validate that the URL belongs to an authorized domain and is safe to redirect to.
    Returns a sanitized/normalized URL string if valid; otherwise raises ValueError.
    """
    if not is_domain_authorized(url):
        raise ValueError("Unauthorized redirect domain")

    parsed = urlparse(url)
    # Handle schemeless URLs like "example.com/path"
    if not parsed.netloc and parsed.path and "://" not in url:
        parsed = urlparse("//" + url)

    # Enforce allowed schemes
    scheme = (parsed.scheme or "https").lower()
    if scheme not in ("http", "https"):
        raise ValueError("Unsupported URL scheme for redirect")

    # Normalize and rebuild netloc without userinfo
    host = parsed.hostname
    if not host:
        raise ValueError("Invalid URL")
    host = host.rstrip(".").lower()
    try:
        host = host.encode("idna").decode("ascii")
    except Exception:
        pass
    port = parsed.port
    netloc = f"{host}:{port}" if port else host

    # Reconstruct sanitized URL
    safe_url = urlunparse((scheme, netloc, parsed.path or "", parsed.params, parsed.query, parsed.fragment))
    return safe_url
