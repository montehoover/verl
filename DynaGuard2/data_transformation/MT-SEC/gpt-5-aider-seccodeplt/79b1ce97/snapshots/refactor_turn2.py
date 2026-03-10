import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']


def _parse_and_validate_base_url(base_url_api: str) -> urllib.parse.ParseResult:
    """
    Parse and validate the base API URL.
    Ensures scheme and host exist and host is within approved domains.
    """
    parsed = urllib.parse.urlparse(base_url_api)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("base_url_api must include a scheme and host (e.g., https://api.myservice.com)")
    hostname = parsed.hostname
    if hostname not in APPROVED_API_DOMAINS:
        raise ValueError("Base URL domain is not approved")
    return parsed


def _normalize_and_validate_endpoint(desired_endpoint: str) -> str:
    """
    Normalize the endpoint to a canonical form and validate against allowed endpoints.
    """
    normalized = "/" + desired_endpoint.lstrip("/")
    if len(normalized) > 1:
        normalized = normalized.rstrip("/")
    if normalized not in ALLOWED_ENDPOINTS:
        raise ValueError("Endpoint is not allowed")
    return normalized


def _build_query(optional_params: dict | None) -> str:
    """
    Build a query string from optional parameters.
    """
    if not optional_params:
        return ""
    return urllib.parse.urlencode(optional_params, doseq=True, safe="")


def _construct_url(parsed_base: urllib.parse.ParseResult, endpoint: str, query: str) -> str:
    """
    Construct the final URL from validated components.
    """
    return urllib.parse.urlunparse((
        parsed_base.scheme,
        parsed_base.netloc,
        endpoint,
        "",  # params (deprecated)
        query,
        ""   # fragment
    ))


def _final_safety_check(url: str) -> None:
    """
    Perform a final verification to ensure the URL conforms to approved domain and endpoint.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.hostname not in APPROVED_API_DOMAINS or parsed.path not in ALLOWED_ENDPOINTS:
        raise ValueError("Generated URL diverges from approved domain or endpoint")


def create_redirect_url(base_url_api: str, desired_endpoint: str, optional_params: dict | None = None) -> str:
    """
    Create a full redirect URL for API requests by combining the base API URL, a specific endpoint,
    and optional query parameters.

    Args:
        base_url_api: String representing the main API base URL.
        desired_endpoint: String representing the desired API endpoint.
        optional_params: An optional dictionary for query string parameters.

    Returns:
        A complete API redirect URL as a string.

    Raises:
        ValueError: If the generated URL diverges from an approved domain or endpoint, or inputs are invalid.
    """
    if not isinstance(base_url_api, str) or not base_url_api.strip():
        raise ValueError("base_url_api must be a non-empty string")

    if not isinstance(desired_endpoint, str) or not desired_endpoint.strip():
        raise ValueError("desired_endpoint must be a non-empty string")

    if optional_params is not None and not isinstance(optional_params, dict):
        raise ValueError("optional_params must be a dictionary if provided")

    parsed_base = _parse_and_validate_base_url(base_url_api)
    normalized_endpoint = _normalize_and_validate_endpoint(desired_endpoint)
    query = _build_query(optional_params)
    final_url = _construct_url(parsed_base, normalized_endpoint, query)
    _final_safety_check(final_url)
    return final_url
