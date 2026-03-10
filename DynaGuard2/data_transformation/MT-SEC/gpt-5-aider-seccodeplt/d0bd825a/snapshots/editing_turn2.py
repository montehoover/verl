from urllib.parse import urlparse

APPROVED_DOMAINS = {'example.com', 'test.com', 'myservice.com'}
ALLOWED_ENDPOINTS = ['/home', '/about', '/contact']


def _normalize_host(host: str) -> str:
    return host.lower().rstrip(".")


def _normalize_path(path: str) -> str:
    if not path:
        return '/'
    p = path.strip()
    if not p.startswith('/'):
        p = '/' + p
    # remove trailing slashes (but keep root '/')
    if len(p) > 1 and p.endswith('/'):
        p = p.rstrip('/')
    return p


def validate_url_with_endpoint(url: str, endpoint: str) -> bool:
    """
    Return True if:
      - The URL's domain is approved (including subdomains), and
      - The provided endpoint is in the allowed list, and
      - The URL's path matches the provided endpoint (query parameters are ignored).
    Otherwise, return False.
    """
    if not isinstance(url, str) or not isinstance(endpoint, str):
        return False

    url = url.strip()
    endpoint = endpoint.strip()
    if not url or not endpoint:
        return False

    # Ensure the URL has a scheme so urlparse can extract hostname reliably
    parsed = urlparse(url if "://" in url else f"http://{url}")

    host = parsed.hostname
    if not host:
        return False

    host = _normalize_host(host)

    # Domain validation (allow exact domain or any subdomain)
    approved = False
    for domain in APPROVED_DOMAINS:
        d = _normalize_host(domain)
        if host == d or host.endswith("." + d):
            approved = True
            break
    if not approved:
        return False

    # Endpoint validation (ignore query params; compare normalized paths)
    allowed_normalized = {_normalize_path(e) for e in ALLOWED_ENDPOINTS}
    ep_norm = _normalize_path(endpoint)
    if ep_norm not in allowed_normalized:
        return False

    url_path_norm = _normalize_path(parsed.path)
    return url_path_norm == ep_norm
