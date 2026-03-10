from urllib.parse import urlparse

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
