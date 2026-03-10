import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']


def _validate_url_components(base_api_url: str, api_endpoint: str):
    """
    Validates the domain of the base_api_url and the api_endpoint.

    Args:
        base_api_url: The base URL of the API.
        api_endpoint: The specific API endpoint.

    Raises:
        ValueError: If the domain is not in APPROVED_API_DOMAINS or
                    if the endpoint is not in ALLOWED_ENDPOINTS.
    """
    parsed_base_url = urllib.parse.urlparse(base_api_url)
    domain = parsed_base_url.netloc

    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not an approved API domain.")

    if api_endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{api_endpoint}' is not an allowed API endpoint.")


def _build_url_with_query(full_url_path: str, query_options: dict = None) -> str:
    """
    Appends query parameters to a URL path.

    Args:
        full_url_path: The URL path (base + endpoint).
        query_options: An optional dictionary of query parameters.

    Returns:
        The URL path with query parameters appended, if any.
    """
    if query_options:
        query_string = urllib.parse.urlencode(query_options)
        return f"{full_url_path}?{query_string}"
    return full_url_path


def construct_api_redirect(base_api_url: str, api_endpoint: str, query_options: dict = None) -> str:
    """
    Generates a redirect URL for making API requests.

    It combines a specified base URL and endpoint, along with optional query
    parameters, into a complete redirect URL.

    Args:
        base_api_url: A string representing the base address of an API.
        api_endpoint: A string specifying the endpoint of the API.
        query_options: An optional dictionary containing the query parameters.

    Returns:
        A fully constructed API redirect URL as a string.

    Raises:
        ValueError: If the generated URL does not belong to an allowed domain
                    or exceeds the list of allowed endpoints.
    """
    _validate_url_components(base_api_url, api_endpoint)

    # Ensure the base_api_url does not end with a slash if the endpoint starts with one,
    # and vice-versa, to prevent double slashes.
    # urllib.parse.urljoin handles this gracefully.
    full_url_path = urllib.parse.urljoin(base_api_url, api_endpoint)

    redirect_url = _build_url_with_query(full_url_path, query_options)
        
    return redirect_url
