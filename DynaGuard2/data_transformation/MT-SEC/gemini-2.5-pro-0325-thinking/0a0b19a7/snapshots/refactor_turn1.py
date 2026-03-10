import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']

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

    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not an approved API domain.")

    if path not in ALLOWED_ENDPOINTS:
        # Allow for trailing slashes in endpoint definitions
        if not (path.endswith('/') and path[:-1] in ALLOWED_ENDPOINTS) and \
           not (any(allowed_ep.endswith('/') and path == allowed_ep[:-1] for allowed_ep in ALLOWED_ENDPOINTS)):
            raise ValueError(f"Endpoint '{path}' is not an allowed API endpoint.")

    # Add query parameters if any
    if query_params:
        query_string = urllib.parse.urlencode(query_params)
        final_url = f"{url_without_params}?{query_string}"
    else:
        final_url = url_without_params
        
    return final_url
