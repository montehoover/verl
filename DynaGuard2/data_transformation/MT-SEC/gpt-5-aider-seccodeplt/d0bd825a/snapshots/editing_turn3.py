import urllib.parse
from urllib.parse import urlparse
from typing import Mapping, Any, Optional

APPROVED_DOMAINS = {'example.com', 'test.com', 'myservice.com'}
APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']


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


def create_api_redirect_url(base_url_api: str, api_endpoint: str, opts: Optional[Mapping[str, Any]] = None) -> str:
    """
    Construct a redirect URL to a specific API endpoint with optional query parameters.

    Requirements:
      - base_url_api must resolve to a host in APPROVED_API_DOMAINS (exact match).
      - api_endpoint must be in ALLOWED_ENDPOINTS (after normalization).
      - opts, if provided, must be a mapping and is encoded as query parameters.
      - The URL is constructed using urllib.parse utilities.
      - Raises ValueError if any condition is not met.
    """
    # Validate input types and presence
    if not isinstance(base_url_api, str) or not base_url_api.strip():
        raise ValueError("base_url_api must be a non-empty string")
    if not isinstance(api_endpoint, str) or not api_endpoint.strip():
        raise ValueError("api_endpoint must be a non-empty string")
    if opts is not None and not isinstance(opts, Mapping):
        raise ValueError("opts must be a mapping (dict-like) if provided")

    base = base_url_api.strip()

    # Parse base URL (default to https if scheme missing)
    parsed_base = urllib.parse.urlparse(base if "://" in base else f"https://{base}")

    # Disallow credentials in base URL for security
    if parsed_base.username or parsed_base.password:
        raise ValueError("base_url_api must not contain user credentials")

    host = parsed_base.hostname
    if not host:
        raise ValueError("base_url_api must include a valid hostname")

    host_norm = _normalize_host(host)
    approved_hosts = {_normalize_host(d) for d in APPROVED_API_DOMAINS}
    if host_norm not in approved_hosts:
        raise ValueError("base_url_api domain is not approved")

    # Normalize and validate endpoint (must not contain its own query/fragment)
    ep_raw = api_endpoint.strip()
    ep_parts = urllib.parse.urlsplit(ep_raw)
    if ep_parts.scheme or ep_parts.netloc:
        raise ValueError("api_endpoint must be a path, not a full URL")
    if ep_parts.query or ep_parts.fragment:
        raise ValueError("api_endpoint must not include query parameters or fragments")

    ep_norm = _normalize_path(ep_parts.path or ep_raw)
    allowed_eps = {_normalize_path(e) for e in ALLOWED_ENDPOINTS}
    if ep_norm not in allowed_eps:
        raise ValueError("api_endpoint is not allowed")

    # Prepare query string from opts (omit None values)
    query = ""
    if opts:
        filtered = {k: v for k, v in opts.items() if v is not None}
        if filtered:
            query = urllib.parse.urlencode(filtered, doseq=True)

    # Build final URL components
    scheme = parsed_base.scheme or "https"
    netloc = host_norm
    if parsed_base.port:
        netloc = f"{netloc}:{parsed_base.port}"

    # Assemble the final URL
    final_url = urllib.parse.urlunparse((
        scheme,
        netloc,
        ep_norm,
        "",        # params (deprecated; unused)
        query,
        ""         # fragment
    ))

    return final_url
