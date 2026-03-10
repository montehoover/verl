import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']


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

    # Parse the base URL and validate scheme and domain
    parsed_base = urllib.parse.urlparse(base_url_api)
    if not parsed_base.scheme or not parsed_base.netloc:
        raise ValueError("base_url_api must include a scheme and host (e.g., https://api.myservice.com)")

    hostname = parsed_base.hostname  # lowercased host without port
    if hostname not in APPROVED_API_DOMAINS:
        raise ValueError("Base URL domain is not approved")

    # Normalize and validate endpoint
    normalized_endpoint = "/" + desired_endpoint.lstrip("/")
    # Normalize trailing slash for comparison, but keep single slash endpoints intact
    if len(normalized_endpoint) > 1:
        normalized_endpoint = normalized_endpoint.rstrip("/")

    if normalized_endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError("Endpoint is not allowed")

    # Build query string if provided
    query = ""
    if optional_params:
        query = urllib.parse.urlencode(optional_params, doseq=True, safe="")

    # Construct the final URL using the approved base (preserving port if present)
    final_url = urllib.parse.urlunparse((
        parsed_base.scheme,
        parsed_base.netloc,
        normalized_endpoint,
        "",  # params (deprecated)
        query,
        ""   # fragment
    ))

    # Final safety check to ensure the generated URL still conforms
    parsed_final = urllib.parse.urlparse(final_url)
    if parsed_final.hostname not in APPROVED_API_DOMAINS or parsed_final.path not in ALLOWED_ENDPOINTS:
        raise ValueError("Generated URL diverges from approved domain or endpoint")

    return final_url
