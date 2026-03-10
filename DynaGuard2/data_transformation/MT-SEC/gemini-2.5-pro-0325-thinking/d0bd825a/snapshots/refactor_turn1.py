import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']

def create_api_redirect_url(base_url_api: str, api_endpoint: str, opts: dict = None) -> str:
    """
    Generates a redirect URL for making API requests.

    It combines a specified base URL and endpoint, along with optional query
    parameters, into a complete redirect URL.

    Args:
        base_url_api: A string representing the base address of an API.
        api_endpoint: A string specifying the endpoint of the API.
        opts: An optional dictionary containing the query parameters.

    Returns:
        A fully constructed API redirect URL as a string.

    Raises:
        ValueError: If the generated URL does not belong to an allowed domain
                    or uses a non-allowed endpoint.
    """
    parsed_base_url = urllib.parse.urlparse(base_url_api)
    if parsed_base_url.netloc not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{parsed_base_url.netloc}' is not an approved API domain.")

    if api_endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{api_endpoint}' is not an allowed endpoint.")

    # Ensure base_url_api ends with a slash and api_endpoint doesn't start with one
    # to prevent double slashes or missing slashes.
    # However, urljoin handles this gracefully.
    
    full_url_path = urllib.parse.urljoin(base_url_api, api_endpoint)

    if opts:
        query_string = urllib.parse.urlencode(opts)
        redirect_url = f"{full_url_path}?{query_string}"
    else:
        redirect_url = full_url_path
        
    return redirect_url
