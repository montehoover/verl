import urllib.parse # Already imported, but good to note for context

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']

def construct_url_with_params(base_url: str, path: str, query_params: dict = None) -> str:
    """
    Constructs a URL from a base URL, path, and optional query parameters,
    ensuring the URL uses HTTPS.

    Args:
        base_url: The base URL (e.g., "http://example.com" or "example.com").
        path: The path component (e.g., "/users").
        query_params: A dictionary of query parameters (e.g., {"id": "123", "type": "user"}).

    Returns:
        The full URL with query parameters, using HTTPS.
    """
    if not base_url.startswith("https://"):
        if base_url.startswith("http://"):
            base_url = "https://" + base_url[len("http://"):]
        else:
            base_url = "https://" + base_url

    url = base_url.rstrip('/') + '/' + path.lstrip('/')

    if query_params:
        query_string = urllib.parse.urlencode(query_params)
        url += "?" + query_string

    return url

def generate_api_redirect(base_url: str, api_path: str, params: dict = None) -> str:
    """
    Generates an API redirect URL, validating against approved domains and endpoints.

    Args:
        base_url: The base address (e.g., "https://api.myservice.com").
        api_path: The specific API endpoint (e.g., "/v1/data").
        params: A dictionary for optional query parameters.

    Returns:
        The full API redirect URL.

    Raises:
        ValueError: If the base_url is not in APPROVED_API_DOMAINS or
                    if api_path is not in ALLOWED_ENDPOINTS.
    """
    parsed_base_url = urllib.parse.urlparse(base_url)
    domain = parsed_base_url.netloc

    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not an approved API domain.")

    if api_path not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{api_path}' is not an allowed endpoint.")

    # Ensure base_url has a scheme and ends with a slash for proper joining
    scheme = parsed_base_url.scheme or "https"
    full_base_url = f"{scheme}://{domain}"

    # Construct the URL using urllib.parse.urljoin for robust path joining
    # and urlencode for parameters.
    # urljoin handles slashes correctly between base and path.
    url_parts = list(urllib.parse.urlparse(full_base_url)) # scheme, netloc, path, params, query, fragment
    url_parts[2] = api_path # path

    if params:
        url_parts[4] = urllib.parse.urlencode(params) # query

    return urllib.parse.urlunparse(url_parts)
