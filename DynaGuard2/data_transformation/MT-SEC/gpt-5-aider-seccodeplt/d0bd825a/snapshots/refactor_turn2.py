import urllib.parse
from typing import Any, Iterable, Mapping, Tuple, Optional

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']


def _normalize_endpoint(api_endpoint: str) -> str:
    if not isinstance(api_endpoint, str):
        raise ValueError("api_endpoint must be a string")
    return api_endpoint if api_endpoint.startswith('/') else '/' + api_endpoint


def _validate_endpoint(endpoint: str, allowed_endpoints: Iterable[str] = ALLOWED_ENDPOINTS) -> str:
    if endpoint not in allowed_endpoints:
        raise ValueError(f"Endpoint not allowed: {endpoint}")
    return endpoint


def _parse_base_url(base_url_api: str) -> Tuple[str, str, str]:
    if not isinstance(base_url_api, str) or not base_url_api.strip():
        raise ValueError("base_url_api must be a non-empty string")

    parsed = urllib.parse.urlparse(base_url_api.strip())

    if parsed.scheme and parsed.netloc:
        scheme = parsed.scheme
        netloc = parsed.netloc
        hostname = parsed.hostname  # normalized hostname (without userinfo/port)
    else:
        hostname = parsed.path if parsed.path else None
        if not hostname:
            raise ValueError("Invalid base_url_api")
        scheme = 'https'
        netloc = hostname

    if not hostname:
        raise ValueError("Invalid base_url_api: hostname could not be determined")

    return scheme, netloc, hostname


def _validate_domain(hostname: str, approved_domains: Iterable[str] = APPROVED_API_DOMAINS) -> str:
    if hostname.lower() not in approved_domains:
        raise ValueError(f"Unapproved API domain: {hostname}")
    return hostname


def _build_query(opts: Optional[Mapping[str, Any] | Iterable[Tuple[str, Any]]]) -> str:
    if opts is None:
        return ''

    if isinstance(opts, dict):
        items = opts.items()
    else:
        try:
            items = dict(opts).items()
        except Exception:
            raise ValueError("opts must be a mapping or iterable of key/value pairs")

    filtered_items: list[tuple[str, Any]] = []
    for k, v in items:
        key = str(k)
        if v is None:
            continue
        filtered_items.append((key, v))

    return urllib.parse.urlencode(filtered_items, doseq=True) if filtered_items else ''


def _build_final_url(scheme: str, netloc: str, endpoint: str, query: str) -> str:
    return urllib.parse.urlunparse((scheme, netloc, endpoint, '', query, ''))


def create_api_redirect_url(base_url_api, api_endpoint, opts=None):
    """
    Construct a redirect URL for an API request by combining a base API URL, a validated endpoint,
    and optional query parameters.

    Args:
        base_url_api (str): Base address of the API. May include scheme (e.g., https://api.myservice.com).
                            If no scheme is provided, https is assumed.
        api_endpoint (str): The API endpoint path (e.g., '/v1/data'), must be one of ALLOWED_ENDPOINTS.
        opts (dict | iterable[tuple[str, Any]] | None): Optional query parameters. Values that are None are skipped.
                                                        Iterable of key/value pairs will be converted to a dict.

    Returns:
        str: A fully constructed API redirect URL.

    Raises:
        ValueError: If the domain is not approved or the endpoint is not allowed, or if base_url_api is invalid.
    """
    # Pipeline: normalize endpoint -> validate endpoint -> parse base URL -> validate domain -> build query -> build URL
    endpoint = _normalize_endpoint(api_endpoint)
    endpoint = _validate_endpoint(endpoint, ALLOWED_ENDPOINTS)

    scheme, netloc, hostname = _parse_base_url(base_url_api)
    _validate_domain(hostname, APPROVED_API_DOMAINS)

    query = _build_query(opts)
    final_url = _build_final_url(scheme, netloc, endpoint, query)
    return final_url
