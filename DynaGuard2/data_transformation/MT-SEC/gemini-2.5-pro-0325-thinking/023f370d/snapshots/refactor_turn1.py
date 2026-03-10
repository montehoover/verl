import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']

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
    parsed_base_url = urllib.parse.urlparse(base_api_url)
    domain = parsed_base_url.netloc
    
    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not an approved API domain.")

    if api_endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{api_endpoint}' is not an allowed API endpoint.")

    # Ensure the base_api_url does not end with a slash if the endpoint starts with one,
    # and vice-versa, to prevent double slashes.
    # urllib.parse.urljoin handles this gracefully.
    full_url_path = urllib.parse.urljoin(base_api_url, api_endpoint)

    if query_options:
        query_string = urllib.parse.urlencode(query_options)
        redirect_url = f"{full_url_path}?{query_string}"
    else:
        redirect_url = full_url_path
        
    return redirect_url
