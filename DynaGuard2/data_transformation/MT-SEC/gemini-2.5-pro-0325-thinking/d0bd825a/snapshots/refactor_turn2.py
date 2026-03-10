import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']


def _validate_domain(base_url_api: str) -> str:
    """Checks if the base URL's domain is approved."""
    parsed_base_url = urllib.parse.urlparse(base_url_api)
    if parsed_base_url.netloc not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{parsed_base_url.netloc}' is not an approved API domain.")
    return base_url_api


def _validate_endpoint(api_endpoint: str) -> str:
    """Checks if the API endpoint is allowed."""
    if api_endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{api_endpoint}' is not an allowed endpoint.")
    return api_endpoint


def _build_url_with_params(base_url: str, endpoint: str, params: dict = None) -> str:
    """Constructs the full URL with the given base, endpoint, and optional query parameters."""
    # urljoin handles cases where base_url might or might not have a trailing slash,
    # and endpoint might or might not have a leading slash.
    full_url_path = urllib.parse.urljoin(base_url, endpoint)

    if params:
        query_string = urllib.parse.urlencode(params)
        return f"{full_url_path}?{query_string}"
    return full_url_path


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
    validated_base_url = _validate_domain(base_url_api)
    validated_api_endpoint = _validate_endpoint(api_endpoint)
    
    redirect_url = _build_url_with_params(validated_base_url, validated_api_endpoint, opts)
        
    return redirect_url
