import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']

def generate_api_redirect(base_url: str, api_path: str, params: dict = None) -> str:
    """
    Generates a redirect URL for making API requests.

    Combines a specified base URL and endpoint, along with optional query
    parameters, into a complete redirect URL.

    Args:
        base_url: A string representing the base address of an API.
        api_path: A string specifying the endpoint of the API.
        params: An optional dictionary containing the query parameters.

    Returns:
        A fully constructed API redirect URL as a string.

    Raises:
        ValueError: If the generated URL does not belong to an allowed domain
                    or uses a disallowed endpoint.
    """
    parsed_base_url = urllib.parse.urlparse(base_url)
    if parsed_base_url.hostname not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{parsed_base_url.hostname}' is not an approved API domain.")

    if api_path not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{api_path}' is not an allowed API endpoint.")

    # Ensure api_path starts with a slash if not already, and base_url doesn't end with one
    # to prevent double slashes when joining. urljoin handles this well.
    full_path = urllib.parse.urljoin(base_url if base_url.endswith('/') else base_url + '/',
                                     api_path.lstrip('/'))

    if params:
        query_string = urllib.parse.urlencode(params)
        return f"{full_path}?{query_string}"
    else:
        return full_path
