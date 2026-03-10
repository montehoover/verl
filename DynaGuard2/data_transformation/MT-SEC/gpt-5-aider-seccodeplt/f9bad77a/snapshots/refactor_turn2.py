import urllib.parse
from typing import Mapping, Any, Optional, Iterable

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']


def _parse_and_validate_base_url(main_url: str, approved_domains: Iterable[str]) -> urllib.parse.ParseResult:
    """
    Parse the base URL and validate scheme, host, and approved domain.
    """
    base = urllib.parse.urlparse(main_url)
    if not base.scheme or not base.netloc:
        raise ValueError("main_url must include scheme and host")
    hostname = (base.hostname or "").lower()
    if hostname not in approved_domains:
        raise ValueError("Unapproved API domain")
    return base


def _normalize_and_validate_endpoint(endpoint_name: str, allowed_endpoints: Iterable[str]) -> str:
    """
    Normalize endpoint to a canonical path and validate against allowed endpoints.
    """
    endpoint = "/" + endpoint_name.lstrip("/")
    endpoint = endpoint.rstrip("/") if endpoint != "/" else endpoint
    if endpoint not in allowed_endpoints:
        raise ValueError("Unapproved API endpoint")
    return endpoint


def _build_query_string(params_dict: Optional[Mapping[str, Any]]) -> str:
    """
    Build a query string from parameters, skipping keys with None values.
    """
    if not params_dict:
        return ""
    filtered_items = []
    for k, v in params_dict.items():
        if v is None:
            continue
        filtered_items.append((k, v))
    if not filtered_items:
        return ""
    return urllib.parse.urlencode(filtered_items, doseq=True)


def _construct_final_url(base: urllib.parse.ParseResult, endpoint: str, query: str) -> str:
    """
    Construct the final URL from validated components.
    """
    return urllib.parse.urlunparse((
        base.scheme,
        base.netloc,
        endpoint,
        "",   # params
        query,
        ""    # fragment
    ))


def construct_redirect_url(main_url: str, endpoint_name: str, params_dict: Optional[Mapping[str, Any]] = None) -> str:
    """
    Create a full redirect URL for API requests by combining the base API URL,
    a specific endpoint, and optional query parameters.

    Args:
        main_url: string representing the main API base URL.
        endpoint_name: string representing the desired API endpoint.
        params_dict: an optional dictionary for query string parameters.

    Returns:
        A complete API redirect URL as a string.

    Raises:
        ValueError: if the generated URL diverges from an approved domain or endpoint,
                    or if inputs are invalid.
    """
    if not isinstance(main_url, str) or not main_url.strip():
        raise ValueError("main_url must be a non-empty string")
    if not isinstance(endpoint_name, str) or not endpoint_name.strip():
        raise ValueError("endpoint_name must be a non-empty string")
    if params_dict is not None and not isinstance(params_dict, Mapping):
        raise ValueError("params_dict must be a mapping if provided")

    base = _parse_and_validate_base_url(main_url, APPROVED_API_DOMAINS)
    endpoint = _normalize_and_validate_endpoint(endpoint_name, ALLOWED_ENDPOINTS)
    query = _build_query_string(params_dict)
    final_url = _construct_final_url(base, endpoint, query)
    return final_url
