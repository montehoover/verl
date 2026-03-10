import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']


def generate_api_redirect(base_url: str, api_path: str, params: dict | None = None) -> str:
    """
    Generate a redirect URL for making API requests.

    Args:
        base_url: Base address of the API (e.g., "https://api.myservice.com").
        api_path: Endpoint path (e.g., "/v1/data").
        params: Optional dictionary of query parameters.

    Returns:
        Fully constructed API redirect URL as a string.

    Raises:
        ValueError: If the base_url is not in an approved domain or api_path is not allowed.
    """
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("base_url must be a non-empty string")
    if not isinstance(api_path, str) or not api_path.strip():
        raise ValueError("api_path must be a non-empty string")
    if params is not None and not isinstance(params, dict):
        raise ValueError("params must be a dictionary if provided")

    # Ensure base_url has a scheme; default to https if missing
    parsed_base = urllib.parse.urlparse(base_url)
    if not parsed_base.scheme:
        parsed_base = urllib.parse.urlparse("https://" + base_url)

    if not parsed_base.hostname:
        raise ValueError("base_url must include a valid hostname")

    # Validate approved domain (ignore port)
    if parsed_base.hostname not in APPROVED_API_DOMAINS:
        raise ValueError("Base URL domain is not approved")

    # Parse and validate the api_path (must be a pure path, no scheme, host, query, or fragment)
    parsed_path = urllib.parse.urlparse(api_path)
    if parsed_path.scheme or parsed_path.netloc or parsed_path.query or parsed_path.params or parsed_path.fragment:
        raise ValueError("api_path must be a path without scheme, domain, query, parameters, or fragment")

    # Normalize path to have a single leading slash and no trailing slash
    normalized_path = "/" + (parsed_path.path or "").lstrip("/")
    if len(normalized_path) > 1:
        normalized_path = normalized_path.rstrip("/")

    # Validate the endpoint is allowed
    if normalized_path not in ALLOWED_ENDPOINTS:
        raise ValueError("Endpoint is not allowed")

    # Build the query string
    query = ""
    if params:
        # Convert keys and values to strings and use doseq=True to support list values
        normalized_params = {}
        for k, v in params.items():
            if v is None:
                # Skip None values
                continue
            normalized_params[str(k)] = v
        query = urllib.parse.urlencode(normalized_params, doseq=True)

    # Rebuild the full URL using the validated components
    final_url = urllib.parse.urlunparse((
        parsed_base.scheme,
        parsed_base.netloc,
        normalized_path,
        "",     # params (deprecated)
        query,
        ""      # fragment
    ))

    return final_url
