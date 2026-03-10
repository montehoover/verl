from typing import Any, Mapping, Optional
from urllib.parse import urlencode
import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']


def construct_url(base_url: str, path: str) -> str:
    """
    Construct a URL by concatenating the base_url and path.

    Args:
        base_url (str): The base URL.
        path (str): The path to append to the base URL.

    Returns:
        str: The concatenated full URL.
    """
    return f"{base_url}{path}"


def construct_url_with_params(
    base_url: str,
    path: str,
    query_params: Optional[Mapping[str, Any]] = None,
) -> str:
    """
    Construct a full URL by ensuring an https:// base, concatenating the path,
    and appending URL-encoded query parameters.

    Args:
        base_url (str): The base URL.
        path (str): The path to append to the base URL.
        query_params (Optional[Mapping[str, Any]]): Dictionary of query parameters.

    Returns:
        str: The complete URL.
    """
    base = base_url.strip()

    # Ensure https:// prefix
    if not base.startswith("https://"):
        if base.startswith("http://"):
            base = "https://" + base[len("http://"):]
        else:
            base = "https://" + base.lstrip("/")

    # Concatenate base and path with a single slash between them
    normalized_path = path.strip()
    if normalized_path:
        if base.endswith("/"):
            base = base[:-1]
        if normalized_path.startswith("/"):
            full_url = base + normalized_path
        else:
            full_url = f"{base}/{normalized_path}"
    else:
        full_url = base

    # Append query parameters if provided
    if query_params:
        query_string = urlencode(query_params, doseq=True)
        if query_string:
            separator = "&" if "?" in full_url else "?"
            full_url = f"{full_url}{separator}{query_string}"

    return full_url


def generate_api_redirect(
    base_url: str,
    api_path: str,
    params: Optional[Mapping[str, Any]] = None,
) -> str:
    """
    Generate a validated API redirect URL.

    The base_url must belong to APPROVED_API_DOMAINS and the api_path must be one of ALLOWED_ENDPOINTS.
    The URL is constructed using urllib.parse and query params are URL-encoded.

    Args:
        base_url (str): The base API URL (e.g., https://api.myservice.com).
        api_path (str): The allowed API endpoint path (e.g., /v1/data).
        params (Optional[Mapping[str, Any]]): Optional query parameters to include.

    Returns:
        str: The fully constructed and validated URL.

    Raises:
        ValueError: If the base domain is not approved or the endpoint is not allowed.
    """
    if not isinstance(base_url, str) or not isinstance(api_path, str):
        raise ValueError("base_url and api_path must be strings.")

    base = base_url.strip()

    # Normalize base to ensure it parses as a network location
    if "://" not in base:
        base = "https://" + base.lstrip("/")

    parsed_base = urllib.parse.urlparse(base)

    # If parsing still didn't produce a hostname, try treating it as a bare host.
    if not parsed_base.hostname and parsed_base.path:
        parsed_base = urllib.parse.urlparse("https://" + parsed_base.path.lstrip("/"))

    hostname = parsed_base.hostname
    if not hostname or hostname not in APPROVED_API_DOMAINS:
        raise ValueError("Base URL domain is not approved.")

    # Normalize and validate the API path
    path = api_path.strip()
    if not path.startswith("/"):
        path = "/" + path

    if path not in ALLOWED_ENDPOINTS:
        raise ValueError("API endpoint is not allowed.")

    # Build the query string
    query = urllib.parse.urlencode(params or {}, doseq=True)

    # Construct final URL using components, ignoring any path/query/fragment from base
    final_url = urllib.parse.urlunparse(
        (
            parsed_base.scheme or "https",
            parsed_base.netloc or hostname,
            path,
            "",   # params (path parameters) - unused
            query,
            "",   # fragment
        )
    )

    return final_url
