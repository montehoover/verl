import urllib.parse
import posixpath

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']


def _normalize_endpoint(endpoint: str) -> str:
    """
    Normalize the endpoint to a canonical absolute path:
    - Must not be a full URL (no scheme or netloc)
    - Ensure it starts with '/'
    - Remove trailing slash (except for root)
    - Collapse any '.' or '..' segments
    """
    if not isinstance(endpoint, str):
        raise TypeError("endpoint must be a string")

    endpoint = endpoint.strip()
    if not endpoint:
        raise ValueError("endpoint cannot be empty")

    parsed_ep = urllib.parse.urlsplit(endpoint)
    if parsed_ep.scheme or parsed_ep.netloc:
        raise ValueError("endpoint must be a path, not a full URL")

    path = parsed_ep.path or "/"
    if not path.startswith("/"):
        path = "/" + path

    # Normalize path to remove duplicate slashes, dots, and dot-dots.
    path = posixpath.normpath(path)
    if not path.startswith("/"):
        # normpath can remove leading slash if path becomes relative
        path = "/" + path

    return path


def build_api_redirect_url(api_base_url, endpoint, query_params=None):
    """
    Construct a redirect URL by combining the base API URL, an endpoint, and optional query parameters.

    Args:
        api_base_url (str): Base URL of the API (e.g., "https://api.myservice.com").
        endpoint (str): Specific API endpoint path (e.g., "/v1/data").
        query_params (dict | None): Optional dict of query parameters.

    Returns:
        str: The complete API redirect URL.

    Raises:
        ValueError: If the resulting URL is not pointing to an approved API domain or endpoint,
                    or if inputs are invalid in a way that prevents safe URL construction.
    """
    if not isinstance(api_base_url, str) or not api_base_url.strip():
        raise ValueError("api_base_url must be a non-empty string")

    # Ensure we have a scheme; default to https if omitted.
    base_str = api_base_url.strip()
    if "://" not in base_str:
        base_str = "https://" + base_str

    base_parts = urllib.parse.urlsplit(base_str)
    if not base_parts.hostname:
        raise ValueError("Invalid api_base_url")

    base_domain = base_parts.hostname.lower()
    if base_domain not in APPROVED_API_DOMAINS:
        raise ValueError("Unapproved API domain")

    # Normalize and validate endpoint
    norm_endpoint = _normalize_endpoint(endpoint)
    if norm_endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError("Unapproved API endpoint")

    # Validate and encode query params
    if query_params is None:
        query = ""
    else:
        if not isinstance(query_params, dict):
            raise TypeError("query_params must be a dict if provided")
        query = urllib.parse.urlencode(query_params, doseq=True)

    # Construct final URL
    final_url = urllib.parse.urlunsplit((
        base_parts.scheme or "https",
        base_parts.netloc,  # preserve port if present
        norm_endpoint,
        query,
        ""
    ))

    # Final safety validation of the constructed URL
    final_parts = urllib.parse.urlsplit(final_url)
    final_domain = (final_parts.hostname or "").lower()
    final_path = posixpath.normpath(final_parts.path or "/")
    if final_domain not in APPROVED_API_DOMAINS or final_path not in ALLOWED_ENDPOINTS:
        raise ValueError("Resulting URL is not pointing to an approved API domain or endpoint")

    return final_url
