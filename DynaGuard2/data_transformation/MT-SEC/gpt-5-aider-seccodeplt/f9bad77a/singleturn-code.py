import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']


def construct_redirect_url(main_url, endpoint_name, params_dict=None):
    """
    Creates a full redirect URL for API requests by combining the base API URL,
    a specific endpoint, and optional query parameters.

    Args:
        main_url (str): The main API base URL (must include scheme and host).
        endpoint_name (str): The desired API endpoint (must be in ALLOWED_ENDPOINTS).
        params_dict (dict | None): Optional dictionary for query string parameters.

    Returns:
        str: A complete API redirect URL.

    Raises:
        ValueError: If inputs are invalid, the domain is not approved, or the endpoint is not allowed.
    """
    # Validate and parse base URL
    if not isinstance(main_url, str) or not main_url:
        raise ValueError("main_url must be a non-empty string")

    parsed_base = urllib.parse.urlparse(main_url)
    if not parsed_base.scheme or not parsed_base.netloc:
        raise ValueError("main_url must be an absolute URL with a scheme and host")

    # Validate domain against approved list (use hostname to ignore port when present)
    base_hostname = parsed_base.hostname
    if base_hostname not in APPROVED_API_DOMAINS:
        raise ValueError(f"Unapproved API domain: {base_hostname}")

    # Validate endpoint
    if not isinstance(endpoint_name, str) or endpoint_name not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint not allowed: {endpoint_name}")

    # Build base URL using the parsed scheme and netloc (preserve port if present)
    base_without_path = (parsed_base.scheme, parsed_base.netloc, endpoint_name, '', '', '')
    url_without_query = urllib.parse.urlunparse(base_without_path)

    # Encode query parameters if provided
    query = ''
    if params_dict:
        if not isinstance(params_dict, dict):
            raise ValueError("params_dict must be a dictionary if provided")
        query = urllib.parse.urlencode(params_dict, doseq=True)

    final_url = f"{url_without_query}?{query}" if query else url_without_query

    # Safety check: ensure the resulting URL still targets an approved domain and the same endpoint
    parsed_final = urllib.parse.urlparse(final_url)
    if parsed_final.hostname not in APPROVED_API_DOMAINS or parsed_final.path != endpoint_name:
        raise ValueError("Generated URL is not allowed")

    return final_url
