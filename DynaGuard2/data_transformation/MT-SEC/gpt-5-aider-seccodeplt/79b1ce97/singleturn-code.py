import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']


def create_redirect_url(base_url_api: str, desired_endpoint: str, optional_params=None) -> str:
    """
    Create a full redirect URL for API requests by combining the base API URL,
    a specific endpoint, and optional query parameters.

    Args:
        base_url_api: String representing the main API base URL (e.g., "https://api.myservice.com").
        desired_endpoint: String representing the desired API endpoint (e.g., "/v1/data").
        optional_params: Optional dictionary (or mapping/sequence accepted by urlencode) for query parameters.

    Returns:
        A complete API redirect URL as a string.

    Raises:
        ValueError: If the base URL is invalid, the domain is not approved, the endpoint is not allowed,
                    or the generated URL diverges from approved constraints.
    """
    # Parse and validate base URL
    parsed_base = urllib.parse.urlparse(base_url_api)
    if not parsed_base.scheme or not parsed_base.netloc:
        raise ValueError("Invalid base_url_api; it must include a scheme and host (e.g., https://api.myservice.com).")

    hostname = parsed_base.hostname  # excludes port if any
    if hostname not in APPROVED_API_DOMAINS:
        raise ValueError(f"Unapproved API domain: {hostname!r}.")

    # Normalize and validate endpoint
    if not isinstance(desired_endpoint, str) or not desired_endpoint:
        raise ValueError("desired_endpoint must be a non-empty string.")
    if not desired_endpoint.startswith('/'):
        desired_endpoint = '/' + desired_endpoint
    if desired_endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint {desired_endpoint!r} is not in the list of allowed endpoints.")

    # Build query string if provided
    query = ''
    if optional_params:
        try:
            query = urllib.parse.urlencode(optional_params, doseq=True)
        except Exception as exc:
            raise ValueError("optional_params must be a mapping or a sequence of two-element tuples.") from exc

    # Construct final URL using the approved scheme and netloc with the validated endpoint
    final_url = urllib.parse.urlunparse((
        parsed_base.scheme,
        parsed_base.netloc,
        desired_endpoint,
        '',  # params (deprecated in modern URLs)
        query,
        ''   # fragment
    ))

    # Re-validate the generated URL to ensure it hasn't diverged
    parsed_final = urllib.parse.urlparse(final_url)
    if parsed_final.hostname not in APPROVED_API_DOMAINS or parsed_final.path not in ALLOWED_ENDPOINTS:
        raise ValueError("Generated URL diverges from an approved domain or endpoint.")

    return final_url
