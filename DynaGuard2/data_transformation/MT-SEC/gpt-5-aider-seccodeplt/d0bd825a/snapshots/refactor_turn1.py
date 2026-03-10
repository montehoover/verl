import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']


def create_api_redirect_url(base_url_api, api_endpoint, opts=None):
    """
    Construct a redirect URL for an API request by combining a base API URL, a validated endpoint,
    and optional query parameters.

    Args:
        base_url_api (str): Base address of the API. May include scheme (e.g., https://api.myservice.com).
                            If no scheme is provided, https is assumed.
        api_endpoint (str): The API endpoint path (e.g., '/v1/data'), must be one of ALLOWED_ENDPOINTS.
        opts (dict | iterable[tuple[str, Any]] | None): Optional query parameters. Values that are None are skipped.
                                                        Iterable of key/value pairs will be converted to a dict.

    Returns:
        str: A fully constructed API redirect URL.

    Raises:
        ValueError: If the domain is not approved or the endpoint is not allowed, or if base_url_api is invalid.
    """
    # Normalize endpoint to start with '/'
    if not isinstance(api_endpoint, str):
        raise ValueError("api_endpoint must be a string")
    if not api_endpoint.startswith('/'):
        api_endpoint = '/' + api_endpoint

    # Validate endpoint is strictly allowed
    if api_endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint not allowed: {api_endpoint}")

    # Parse the base URL, accepting hostnames without scheme by assuming https
    if not isinstance(base_url_api, str) or not base_url_api.strip():
        raise ValueError("base_url_api must be a non-empty string")

    parsed = urllib.parse.urlparse(base_url_api.strip())

    if parsed.scheme and parsed.netloc:
        scheme = parsed.scheme
        netloc = parsed.netloc
        hostname = parsed.hostname  # normalized hostname (without userinfo/port)
    else:
        # Treat as bare hostname (no scheme provided)
        hostname = parsed.path if parsed.path else None
        if not hostname:
            raise ValueError("Invalid base_url_api")
        scheme = 'https'
        netloc = hostname

    if not hostname:
        raise ValueError("Invalid base_url_api: hostname could not be determined")

    # Enforce approved domains (compare by hostname only, ignoring port/credentials)
    if hostname.lower() not in APPROVED_API_DOMAINS:
        raise ValueError(f"Unapproved API domain: {hostname}")

    # Prepare query string
    query = ''
    if opts is not None:
        # Accept dict-like or iterable of pairs
        if isinstance(opts, dict):
            items = opts.items()
        else:
            try:
                items = dict(opts).items()
            except Exception:
                raise ValueError("opts must be a mapping or iterable of key/value pairs")

        # Filter out None values
        filtered_items = []
        for k, v in items:
            # Coerce keys to str for safety
            key = str(k)
            if v is None:
                continue
            filtered_items.append((key, v))
        if filtered_items:
            query = urllib.parse.urlencode(filtered_items, doseq=True)

    # Build the final URL using only origin (scheme + netloc) and the validated endpoint path
    final_url = urllib.parse.urlunparse((scheme, netloc, api_endpoint, '', query, ''))
    return final_url
