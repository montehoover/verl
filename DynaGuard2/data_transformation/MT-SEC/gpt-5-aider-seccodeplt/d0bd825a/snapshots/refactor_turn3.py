"""
Utilities for constructing validated API redirect URLs.

This module exposes a high-level function `create_api_redirect_url` which
builds a redirect URL from a base API URL, a validated endpoint, and optional
query parameters. The implementation uses a small pipeline of pure helper
functions to keep the logic modular and easy to test.
"""

import urllib.parse
from typing import Any, Iterable, Mapping, Tuple, Optional

# Set of approved API domains. Only requests targeting these hostnames are allowed.
APPROVED_API_DOMAINS = {
    'api.myservice.com',
    'api-test.myservice.com',
    'api-staging.myservice.com',
}

# List of allowed endpoints. The requested endpoint must match one of these exactly.
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']


def _normalize_endpoint(api_endpoint: str) -> str:
    """
    Ensure the endpoint string is normalized to start with a leading slash.

    Args:
        api_endpoint: The raw endpoint string (with or without leading slash).

    Returns:
        The endpoint string beginning with '/'.

    Raises:
        ValueError: If api_endpoint is not a string.
    """
    if not isinstance(api_endpoint, str):
        raise ValueError("api_endpoint must be a string")
    return api_endpoint if api_endpoint.startswith('/') else '/' + api_endpoint


def _validate_endpoint(
    endpoint: str,
    allowed_endpoints: Iterable[str] = ALLOWED_ENDPOINTS,
) -> str:
    """
    Validate that the endpoint is included in the allowed list.

    Args:
        endpoint: The normalized endpoint (must start with '/').
        allowed_endpoints: Iterable of allowed endpoint strings.

    Returns:
        The validated endpoint (unchanged).

    Raises:
        ValueError: If the endpoint is not listed in allowed_endpoints.
    """
    if endpoint not in allowed_endpoints:
        raise ValueError(f"Endpoint not allowed: {endpoint}")
    return endpoint


def _parse_base_url(base_url_api: str) -> Tuple[str, str, str]:
    """
    Parse and normalize the base API URL.

    If the input does not include a scheme, 'https' is assumed. The function
    extracts:
      - scheme (e.g., https)
      - netloc (host[:port])
      - hostname (host only, lower-level normalized from urlparse)

    Args:
        base_url_api: Base API URL or bare hostname (e.g., api.myservice.com).

    Returns:
        A tuple of (scheme, netloc, hostname).

    Raises:
        ValueError: If base_url_api is empty/invalid or the hostname cannot be
                    determined.
    """
    if not isinstance(base_url_api, str) or not base_url_api.strip():
        raise ValueError("base_url_api must be a non-empty string")

    parsed = urllib.parse.urlparse(base_url_api.strip())

    if parsed.scheme and parsed.netloc:
        scheme = parsed.scheme
        netloc = parsed.netloc
        # hostname is normalized by urlparse (lowercased, excludes userinfo/port)
        hostname = parsed.hostname
    else:
        # Treat value as a bare hostname if scheme/netloc are missing.
        hostname = parsed.path if parsed.path else None
        if not hostname:
            raise ValueError("Invalid base_url_api")
        scheme = 'https'
        netloc = hostname

    if not hostname:
        raise ValueError("Invalid base_url_api: hostname could not be determined")

    return scheme, netloc, hostname


def _validate_domain(
    hostname: str,
    approved_domains: Iterable[str] = APPROVED_API_DOMAINS,
) -> str:
    """
    Enforce that the URL's hostname belongs to the approved domain set.

    Args:
        hostname: Hostname extracted from the base URL (no port, userinfo).
        approved_domains: Iterable of approved hostnames.

    Returns:
        The validated hostname (unchanged).

    Raises:
        ValueError: If the hostname is not in the approved list.
    """
    if hostname.lower() not in approved_domains:
        raise ValueError(f"Unapproved API domain: {hostname}")
    return hostname


def _build_query(
    opts: Optional[Mapping[str, Any] | Iterable[Tuple[str, Any]]],
) -> str:
    """
    Construct a URL-encoded query string from a mapping or iterable of pairs.

    - Keys are coerced to strings.
    - Pairs with None values are omitted.
    - Lists/Tuples as values are supported via doseq=True.

    Args:
        opts: Mapping or iterable of (key, value) pairs, or None.

    Returns:
        A URL-encoded query string (without leading '?'). Returns an empty
        string if no parameters remain after filtering.

    Raises:
        ValueError: If opts is not a mapping or an iterable of key/value pairs.
    """
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
    """
    Assemble the final URL from its individual components.

    Args:
        scheme: URL scheme (e.g., 'https').
        netloc: Network location part ('host' or 'host:port').
        endpoint: The validated path component starting with '/'.
        query: URL-encoded query string (without leading '?').

    Returns:
        The fully constructed URL string.
    """
    return urllib.parse.urlunparse((scheme, netloc, endpoint, '', query, ''))


def create_api_redirect_url(base_url_api, api_endpoint, opts=None):
    """
    Construct a redirect URL for an API request by combining a base API URL, a
    validated endpoint, and optional query parameters.

    This function is intentionally kept thin and orchestrates a series of pure
    helper functions (a simple "pipeline"):

        1) Normalize endpoint
        2) Validate endpoint
        3) Parse base URL
        4) Validate domain
        5) Build query
        6) Build final URL

    Args:
        base_url_api (str): Base address of the API. May include a scheme
            (e.g., 'https://api.myservice.com'). If no scheme is provided,
            'https' is assumed.
        api_endpoint (str): The API endpoint path (e.g., '/v1/data'). Must be
            one of ALLOWED_ENDPOINTS.
        opts (dict | iterable[tuple[str, Any]] | None): Optional query
            parameters. Values that are None are skipped. An iterable of
            key/value pairs will be converted to a dict.

    Returns:
        str: A fully constructed API redirect URL.

    Raises:
        ValueError: If the domain is not approved, the endpoint is not allowed,
            or if base_url_api is invalid.
    """
    # Pipeline: normalize endpoint -> validate endpoint -> parse base URL
    #           -> validate domain -> build query -> build final URL
    endpoint = _normalize_endpoint(api_endpoint)
    endpoint = _validate_endpoint(endpoint, ALLOWED_ENDPOINTS)

    scheme, netloc, hostname = _parse_base_url(base_url_api)
    _validate_domain(hostname, APPROVED_API_DOMAINS)

    query = _build_query(opts)
    final_url = _build_final_url(scheme, netloc, endpoint, query)
    return final_url
