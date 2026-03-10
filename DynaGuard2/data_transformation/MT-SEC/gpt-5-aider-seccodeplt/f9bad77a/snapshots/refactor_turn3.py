"""
Utilities for constructing safe redirect URLs for approved API endpoints.

This module exposes `construct_redirect_url`, which creates a full API redirect
URL from a base URL, an endpoint name, and optional query parameters. The
implementation uses a small pipeline of pure helper functions for improved
readability and testability.
"""

import urllib.parse
from typing import Mapping, Any, Optional, Iterable


# Approved domains and allowed endpoints for constructed redirect URLs.
APPROVED_API_DOMAINS = {
    'api.myservice.com',
    'api-test.myservice.com',
    'api-staging.myservice.com',
}

ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']


def _parse_and_validate_base_url(
    main_url: str,
    approved_domains: Iterable[str],
) -> urllib.parse.ParseResult:
    """
    Parse the base URL and validate scheme, host, and approved domain.

    Args:
        main_url: The candidate base URL, including scheme and host.
        approved_domains: Iterable of domain names that are permitted.

    Returns:
        A urllib.parse.ParseResult representing the parsed base URL.

    Raises:
        ValueError: If the scheme or host is missing, or the host is not in
            the approved domains list.
    """
    # Parse the base URL and check that it has a scheme and host component.
    base = urllib.parse.urlparse(main_url)
    if not base.scheme or not base.netloc:
        raise ValueError("main_url must include scheme and host")

    # Normalize the hostname and ensure it is approved.
    hostname = (base.hostname or "").lower()
    if hostname not in approved_domains:
        raise ValueError("Unapproved API domain")

    return base


def _normalize_and_validate_endpoint(
    endpoint_name: str,
    allowed_endpoints: Iterable[str],
) -> str:
    """
    Normalize endpoint to a canonical path and validate against allowed endpoints.

    Args:
        endpoint_name: The requested endpoint, possibly with or without a
            leading slash and possibly with a trailing slash.
        allowed_endpoints: Iterable of endpoint paths that are permitted.

    Returns:
        The normalized, canonical endpoint path (always begins with a single slash).

    Raises:
        ValueError: If the normalized endpoint is not in the allowed endpoints.
    """
    # Ensure a single leading slash, and remove a trailing slash (except for root).
    endpoint = "/" + endpoint_name.lstrip("/")
    endpoint = endpoint.rstrip("/") if endpoint != "/" else endpoint

    # Validate endpoint against the allowlist.
    if endpoint not in allowed_endpoints:
        raise ValueError("Unapproved API endpoint")

    return endpoint


def _build_query_string(params_dict: Optional[Mapping[str, Any]]) -> str:
    """
    Build a URL-encoded query string from parameters.

    Skips keys whose values are None and supports encoding iterables via doseq.

    Args:
        params_dict: Mapping of query parameter names to values. Values may be
            scalars or sequences.

    Returns:
        A URL-encoded query string (without the leading '?'). Returns an empty
        string if there are no effective parameters to include.
    """
    # Fast path when there are no parameters.
    if not params_dict:
        return ""

    # Filter out keys with None values.
    filtered_items = []
    for key, value in params_dict.items():
        if value is None:
            continue
        filtered_items.append((key, value))

    if not filtered_items:
        return ""

    # Encode parameters; doseq=True expands sequences into multiple key/value pairs.
    return urllib.parse.urlencode(filtered_items, doseq=True)


def _construct_final_url(
    base: urllib.parse.ParseResult,
    endpoint: str,
    query: str,
) -> str:
    """
    Construct the final URL from validated components.

    Args:
        base: The parsed base URL (scheme and netloc are used).
        endpoint: The validated endpoint path.
        query: The URL-encoded query string (without the leading '?').

    Returns:
        The final, fully constructed URL as a string.
    """
    # Build the URL using only validated components.
    return urllib.parse.urlunparse((
        base.scheme,
        base.netloc,
        endpoint,
        "",   # params
        query,
        "",   # fragment
    ))


def construct_redirect_url(
    main_url: str,
    endpoint_name: str,
    params_dict: Optional[Mapping[str, Any]] = None,
) -> str:
    """
    Create a full redirect URL for API requests.

    Combines the base API URL, a specific endpoint, and optional query parameters,
    enforcing allowlists for host and endpoint to prevent unsafe redirects.

    Args:
        main_url: String representing the main API base URL.
        endpoint_name: String representing the desired API endpoint.
        params_dict: Optional mapping of query string parameters.

    Returns:
        A complete API redirect URL as a string.

    Raises:
        ValueError: If inputs are invalid or the generated URL diverges from an
            approved domain or endpoint.
    """
    # Basic input validation for types and non-empty values.
    if not isinstance(main_url, str) or not main_url.strip():
        raise ValueError("main_url must be a non-empty string")
    if not isinstance(endpoint_name, str) or not endpoint_name.strip():
        raise ValueError("endpoint_name must be a non-empty string")
    if params_dict is not None and not isinstance(params_dict, Mapping):
        raise ValueError("params_dict must be a mapping if provided")

    # Pipeline: parse/validate base URL -> normalize/validate endpoint
    #         -> build query string -> construct final URL.
    base = _parse_and_validate_base_url(main_url, APPROVED_API_DOMAINS)
    endpoint = _normalize_and_validate_endpoint(endpoint_name, ALLOWED_ENDPOINTS)
    query = _build_query_string(params_dict)
    final_url = _construct_final_url(base, endpoint, query)

    return final_url
