import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']


def create_api_redirect_url(base_url_api: str, api_endpoint: str, opts: dict | None = None) -> str:
    """
    Construct a redirect URL for API requests using a base URL, an allowed endpoint, and optional query parameters.

    Args:
        base_url_api: Base address of the API (e.g., "https://api.myservice.com").
        api_endpoint: API endpoint path (must be one of ALLOWED_ENDPOINTS).
        opts: Optional dictionary of query parameters.

    Returns:
        Fully constructed API redirect URL as a string.

    Raises:
        ValueError: If domain is not approved, endpoint is not allowed, or inputs are invalid.
    """
    # Basic input validation
    if not isinstance(base_url_api, str) or not base_url_api.strip():
        raise ValueError("base_url_api must be a non-empty string")
    if not isinstance(api_endpoint, str) or not api_endpoint.strip():
        raise ValueError("api_endpoint must be a non-empty string")
    if opts is not None and not isinstance(opts, dict):
        raise ValueError("opts must be a dict if provided")

    base = base_url_api.strip()

    # Ensure URL has a scheme; default to https if omitted
    if "://" not in base:
        base = "https://" + base

    parsed_base = urllib.parse.urlparse(base)
    if not parsed_base.netloc:
        raise ValueError("Invalid base_url_api")

    # Validate domain against the approved set (ignore port and userinfo if present)
    host = parsed_base.netloc.split('@')[-1].split(':')[0]
    if host not in APPROVED_API_DOMAINS:
        raise ValueError(f"Base URL domain '{host}' is not in the approved list")

    # Normalize and validate endpoint
    endpoint_path = "/" + api_endpoint.strip().lstrip("/")
    if endpoint_path not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{endpoint_path}' is not allowed")

    # Build query string from opts
    query = ""
    if opts:
        normalized_opts: dict[str, object] = {}
        for k, v in opts.items():
            key = str(k)
            if isinstance(v, (list, tuple)):
                normalized_opts[key] = [str(x) for x in v]
            else:
                normalized_opts[key] = str(v)
        query = urllib.parse.urlencode(normalized_opts, doseq=True)

    # Construct the final URL
    full_url = urllib.parse.urlunparse((
        parsed_base.scheme or "https",
        parsed_base.netloc,
        endpoint_path,
        "",
        query,
        ""
    ))

    # Final validation of constructed URL
    parsed_final = urllib.parse.urlparse(full_url)
    final_host = parsed_final.netloc.split('@')[-1].split(':')[0]
    if final_host not in APPROVED_API_DOMAINS:
        raise ValueError(f"Resulting URL domain '{final_host}' is not approved")
    if parsed_final.path not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Resulting URL endpoint '{parsed_final.path}' is not allowed")

    return full_url
