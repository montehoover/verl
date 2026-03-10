import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']


def _validate_url_components(domain: str, path: str, approved_domains: set, allowed_endpoints: list) -> None:
    """
    Validates the domain and path of a URL.

    Args:
        domain: The domain of the URL.
        path: The path of the URL.
        approved_domains: A set of approved API domains.
        allowed_endpoints: A list of allowed API endpoints.

    Raises:
        ValueError: If the domain or path is not approved/allowed.
    """
    if domain not in approved_domains:
        raise ValueError(f"Domain '{domain}' is not an approved API domain.")

    if path not in allowed_endpoints:
        # Allow for trailing slashes in endpoint definitions
        if not (path.endswith('/') and path[:-1] in allowed_endpoints) and \
           not (any(allowed_ep.endswith('/') and path == allowed_ep[:-1] for allowed_ep in allowed_endpoints)):
            raise ValueError(f"Endpoint '{path}' is not an allowed API endpoint.")

def _construct_url_with_query_params(base_url: str, query_params: dict = None) -> str:
    """
    Constructs a URL with optional query parameters.

    Args:
        base_url: The base URL without query parameters.
        query_params: An optional dictionary of query parameters.

    Returns:
        The URL with query parameters, if any.
    """
    if query_params:
        query_string = urllib.parse.urlencode(query_params)
        return f"{base_url}?{query_string}"
    return base_url

def build_api_redirect_url(api_base_url: str, endpoint: str, query_params: dict = None) -> str:
    """
    Constructs a redirect URL for API responses.

    Args:
        api_base_url: The base URL of the API.
        endpoint: The specific API endpoint.
        query_params: An optional dictionary of query parameters.

    Returns:
        The complete API redirect URL.

    Raises:
        ValueError: If the resulting URL is not pointing to an approved API domain or endpoint.
    """
    # Ensure the endpoint starts with a slash if it's not empty and api_base_url doesn't end with one
    if endpoint and not endpoint.startswith('/') and not api_base_url.endswith('/'):
        full_path = '/' + endpoint
    elif endpoint.startswith('/') and api_base_url.endswith('/'):
        full_path = endpoint[1:]
    else:
        full_path = endpoint

    # Construct the base URL + endpoint
    url_without_params = urllib.parse.urljoin(api_base_url, full_path)

    # Parse the constructed URL to validate domain and path
    parsed_url = urllib.parse.urlparse(url_without_params)
    domain = parsed_url.netloc
    path = parsed_url.path

    # Validate domain and path
    _validate_url_components(domain, path, APPROVED_API_DOMAINS, ALLOWED_ENDPOINTS)

    # Add query parameters if any
    final_url = _construct_url_with_query_params(url_without_params, query_params)
        
    return final_url
