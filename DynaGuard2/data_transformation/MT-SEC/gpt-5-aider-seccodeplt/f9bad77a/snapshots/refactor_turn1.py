import urllib.parse
from typing import Mapping, Any, Optional

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']


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

    # Parse and validate the base URL
    base = urllib.parse.urlparse(main_url)
    if not base.scheme or not base.netloc:
        raise ValueError("main_url must include scheme and host")
    hostname = (base.hostname or "").lower()
    if hostname not in APPROVED_API_DOMAINS:
        raise ValueError("Unapproved API domain")

    # Normalize and validate endpoint
    endpoint = "/" + endpoint_name.lstrip("/")
    endpoint = endpoint.rstrip("/") if endpoint != "/" else endpoint
    if endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError("Unapproved API endpoint")

    # Build the query string
    query = ""
    if params_dict:
        # Drop keys with None values
        filtered_items = []
        for k, v in params_dict.items():
            if v is None:
                continue
            filtered_items.append((k, v))
        if filtered_items:
            query = urllib.parse.urlencode(filtered_items, doseq=True)

    # Construct final URL strictly using approved host and validated endpoint
    final_url = urllib.parse.urlunparse((
        base.scheme,
        base.netloc,
        endpoint,
        "",  # params
        query,
        ""   # fragment
    ))

    return final_url
