import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']


def create_api_redirect_link(base_url, path, params=None):
    """
    Create an API redirect URL by merging a base API URL with a defined endpoint and optional query parameters.

    Args:
        base_url (str): The API's base URL (e.g., https://api.myservice.com).
        path (str): The target API endpoint (must be one of ALLOWED_ENDPOINTS).
        params (dict, optional): Optional key-value pairs to include as query parameters.

    Returns:
        str: The assembled API URL.

    Raises:
        ValueError: If the base URL is invalid, or if the domain or endpoint is not approved.
    """
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("Invalid base URL")

    # Validate and normalize the path
    if not isinstance(path, str) or not path.strip():
        raise ValueError("Invalid endpoint path")

    # Reject path that attempts to include scheme/netloc/query/fragment
    parsed_path = urllib.parse.urlsplit(path)
    if parsed_path.scheme or parsed_path.netloc or parsed_path.query or parsed_path.fragment:
        raise ValueError("Endpoint path must not include scheme, domain, query, or fragment")

    normalized_path = parsed_path.path if parsed_path.path.startswith('/') else '/' + parsed_path.path

    # Check endpoint authorization
    if normalized_path not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Unapproved endpoint: {normalized_path}")

    # Parse and validate the base URL
    parsed_base = urllib.parse.urlparse(base_url)
    if parsed_base.scheme not in ('http', 'https') or not parsed_base.netloc or parsed_base.hostname is None:
        raise ValueError("Invalid base URL")

    # Ensure the domain is approved
    hostname = parsed_base.hostname.lower()
    if hostname not in APPROVED_API_DOMAINS:
        raise ValueError(f"Unapproved domain: {hostname}")

    # Merge query parameters: start with base URL's query, then overlay with params (params take precedence)
    base_query_map = urllib.parse.parse_qs(parsed_base.query, keep_blank_values=True)

    merged_query_map = {}
    # Normalize base query to list-of-strings
    for k, v_list in base_query_map.items():
        merged_query_map[k] = [str(v) for v in v_list]

    # Overlay provided params (if any)
    if params:
        if not isinstance(params, dict):
            raise ValueError("Params must be a dictionary if provided")
        for k, v in params.items():
            if v is None:
                # Skip None-valued params
                continue
            if isinstance(v, (list, tuple)):
                merged_query_map[k] = [str(item) for item in v]
            else:
                merged_query_map[k] = [str(v)]

    final_query = urllib.parse.urlencode(merged_query_map, doseq=True)

    # Build the final URL explicitly to avoid surprises from urljoin
    final_url = urllib.parse.urlunparse((
        parsed_base.scheme,
        parsed_base.netloc,
        normalized_path,
        '',  # params (deprecated)
        final_query,
        ''   # fragment
    ))

    # Safety check: ensure final host remains approved
    parsed_final = urllib.parse.urlparse(final_url)
    if (parsed_final.hostname or '').lower() not in APPROVED_API_DOMAINS:
        raise ValueError("Constructed URL points to an unapproved domain")

    return final_url
