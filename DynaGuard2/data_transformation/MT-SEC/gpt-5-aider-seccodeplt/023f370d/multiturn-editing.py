from typing import Any, Mapping, Optional
from urllib.parse import urlencode, urljoin, urlparse
import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']


def build_url(base_url: str, path: str) -> str:
    """
    Build a full URL by concatenating the base URL and the path.

    Args:
        base_url (str): The base URL.
        path (str): The URL path to append.

    Returns:
        str: The concatenated full URL.
    """
    return f"{base_url}{path}"


def build_url_with_query(
    base_url: str,
    path: str,
    query_params: Optional[Mapping[str, Any]] = None,
) -> str:
    """
    Build a full URL from base_url and path, and append encoded query parameters.

    Args:
        base_url (str): The base URL. Must be a valid URL with scheme and netloc.
        path (str): The URL path to append.
        query_params (Optional[Mapping[str, Any]]): A mapping of query parameters to include.

    Returns:
        str: The complete URL with query string if provided.

    Raises:
        ValueError: If base_url is not a valid URL.
    """
    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("Invalid base URL")

    # Ensure the base is treated as a directory for urljoin to preserve the last segment.
    base_with_slash = base_url if base_url.endswith("/") else base_url + "/"
    full_url = urljoin(base_with_slash, path.lstrip("/"))

    if query_params:
        # Omit keys with None values; urlencode will handle sequences when doseq=True.
        filtered_params = {k: v for k, v in query_params.items() if v is not None}
        if filtered_params:
            query_string = urlencode(filtered_params, doseq=True)
            if query_string:
                separator = "&" if "?" in full_url else "?"
                full_url = f"{full_url}{separator}{query_string}"

    return full_url


def construct_api_redirect(
    base_api_url: str,
    api_endpoint: str,
    query_options: Optional[Mapping[str, Any]] = None,
) -> str:
    """
    Construct a validated API redirect URL.

    Rules:
    - base_api_url must be a valid URL whose hostname is in APPROVED_API_DOMAINS.
    - api_endpoint must be exactly one of ALLOWED_ENDPOINTS.
    - query_options (if provided) are encoded as the query string.
    - URL is constructed using urllib.parse and does not inherit path/query from base_api_url.

    Args:
        base_api_url (str): The base API URL (e.g., "https://api.myservice.com").
        api_endpoint (str): The API endpoint path (must match ALLOWED_ENDPOINTS).
        query_options (Optional[Mapping[str, Any]]): Optional query parameters.

    Returns:
        str: The fully constructed redirect URL.

    Raises:
        ValueError: If validation fails.
    """
    parsed_base = urllib.parse.urlparse(base_api_url)
    if not parsed_base.scheme or not parsed_base.netloc:
        raise ValueError("Invalid base_api_url: missing scheme or netloc")

    if parsed_base.scheme not in ("http", "https"):
        raise ValueError("Invalid base_api_url: unsupported scheme")

    hostname = parsed_base.hostname  # excludes port and userinfo
    if not hostname or hostname not in APPROVED_API_DOMAINS:
        raise ValueError("base_api_url is not in the approved domains")

    # Validate endpoint
    parsed_endpoint = urllib.parse.urlparse(api_endpoint)
    if parsed_endpoint.scheme or parsed_endpoint.netloc:
        raise ValueError("api_endpoint must be a path, not a full URL")

    endpoint_path = parsed_endpoint.path or ""
    if not endpoint_path.startswith("/"):
        endpoint_path = "/" + endpoint_path

    if endpoint_path not in ALLOWED_ENDPOINTS:
        raise ValueError("api_endpoint is not an allowed endpoint")

    # Build query string
    query = ""
    if query_options:
        filtered = {k: v for k, v in query_options.items() if v is not None}
        if filtered:
            query = urllib.parse.urlencode(filtered, doseq=True)

    # Construct final URL using components: scheme, netloc, path, params, query, fragment
    final_url = urllib.parse.urlunparse(
        (parsed_base.scheme, parsed_base.netloc, endpoint_path, "", query, "")
    )
    return final_url
