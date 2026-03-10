import urllib.parse
import posixpath
import re
from typing import Optional, Mapping, Any

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']


def _normalize_endpoint(endpoint: str) -> str:
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise ValueError("endpoint must be a non-empty string")

    # Disallow absolute URLs or protocol-relative URLs
    if re.match(r'^\w+://', endpoint) or endpoint.startswith('//'):
        raise ValueError("endpoint must be a path, not a full URL")

    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme or parsed.netloc:
        raise ValueError("endpoint must not include a scheme or domain")

    path = parsed.path or ''
    if not path.startswith('/'):
        path = '/' + path

    # Collapse multiple slashes
    path = re.sub(r'/+', '/', path)

    # Normalize dot segments
    normalized = posixpath.normpath(path)

    # Ensure root stays as "/" and others don't have trailing slash
    if normalized != '/' and normalized.endswith('/'):
        normalized = normalized.rstrip('/')

    return normalized


def _normalize_allowed_endpoints(endpoints) -> set:
    normed = set()
    for p in endpoints:
        p = p if isinstance(p, str) else str(p)
        if not p.startswith('/'):
            p = '/' + p
        p = re.sub(r'/+', '/', p)
        p = posixpath.normpath(p)
        if p != '/' and p.endswith('/'):
            p = p.rstrip('/')
        normed.add(p)
    return normed


def build_query_string(base_query: str, query_params: Optional[Mapping[str, Any]]) -> str:
    """
    Construct a query string by merging an existing query from a base URL with provided parameters.

    - Existing parameters from base_query are preserved unless overridden by query_params.
    - None values in query_params are ignored.
    - Iterable values (list/tuple) are encoded with multiple keys via doseq=True.
    """
    merged = urllib.parse.parse_qs(base_query, keep_blank_values=True)

    if query_params:
        for k, v in query_params.items():
            if v is None:
                continue
            key = str(k)
            if isinstance(v, (list, tuple)):
                merged[key] = [str(item) for item in v]
            else:
                merged[key] = [str(v)]

    return urllib.parse.urlencode(merged, doseq=True)


def validate_and_parse_api_base_url(api_base_url: str) -> urllib.parse.ParseResult:
    """
    Parse and validate the API base URL.

    Ensures:
    - Scheme is http or https.
    - Hostname exists and is within APPROVED_API_DOMAINS.
    """
    try:
        base = urllib.parse.urlparse(api_base_url)
    except Exception as e:
        raise ValueError(f"Invalid api_base_url: {e}") from e

    if base.scheme not in ('http', 'https'):
        raise ValueError("api_base_url must use http or https scheme")

    if not base.hostname:
        raise ValueError("api_base_url must include a valid hostname")

    if base.hostname not in APPROVED_API_DOMAINS:
        raise ValueError("api_base_url must point to an approved API domain")

    return base


def validate_resulting_url(final_url: str, base_path: str, normalized_endpoint: str) -> None:
    """
    Validate that the final URL points to an approved domain and exactly the expected endpoint path.
    """
    parsed_final = urllib.parse.urlparse(final_url)

    if parsed_final.hostname not in APPROVED_API_DOMAINS:
        raise ValueError("Resulting URL does not point to an approved API domain")

    expected_path = posixpath.join(base_path or '', normalized_endpoint.lstrip('/'))
    if not expected_path.startswith('/'):
        expected_path = '/' + expected_path

    if posixpath.normpath(parsed_final.path) != posixpath.normpath(expected_path):
        raise ValueError("Resulting URL does not point to an approved API endpoint")


def build_api_redirect_url(api_base_url: str, endpoint: str, query_params: Optional[Mapping[str, Any]] = None) -> str:
    """
    Construct a redirect URL for API responses by combining a base API URL with
    a user-provided endpoint and optional query parameters.

    Args:
        api_base_url: The base URL of the API (including scheme and domain).
        endpoint: The specific API endpoint path (e.g., "/v1/data").
        query_params: Optional dictionary of query parameters.

    Returns:
        A string representing the complete API redirect URL.

    Raises:
        ValueError: If the resulting URL is not pointing to an approved API domain or endpoint,
                    or if inputs are malformed.
    """
    # Parse and validate base URL
    base = validate_and_parse_api_base_url(api_base_url)

    # Normalize and validate endpoint
    normalized_endpoint = _normalize_endpoint(endpoint)
    allowed = _normalize_allowed_endpoints(ALLOWED_ENDPOINTS)
    if normalized_endpoint not in allowed:
        raise ValueError("endpoint is not in the list of allowed API endpoints")

    # Build final path by combining base path and endpoint
    base_path = base.path or ''
    combined_path = posixpath.join(base_path, normalized_endpoint.lstrip('/'))
    if not combined_path.startswith('/'):
        combined_path = '/' + combined_path

    # Merge query params (base query + provided)
    final_query = build_query_string(base.query, query_params)

    # Construct final URL
    final_url = urllib.parse.urlunparse((
        base.scheme,
        base.netloc,
        combined_path,
        '',  # params (deprecated)
        final_query,
        ''   # fragment
    ))

    # Final validation: ensure the URL still points to an approved domain and endpoint
    validate_resulting_url(final_url, base_path, normalized_endpoint)

    return final_url
