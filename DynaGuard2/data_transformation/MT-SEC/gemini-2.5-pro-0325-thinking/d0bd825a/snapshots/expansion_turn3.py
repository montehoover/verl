from urllib.parse import urlparse, urlencode, urlunparse, urljoin

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']

def validate_and_parse_url(url: str) -> dict:
    """
    Validates and parses a URL string.

    Args:
        url: The URL string to validate and parse.

    Returns:
        A dictionary containing the URL's components: scheme, domain, and path.

    Raises:
        ValueError: If the URL is invalid or missing essential components.
    """
    if not isinstance(url, str):
        raise ValueError("Input URL must be a string.")

    parsed_url = urlparse(url)

    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError(f"Invalid URL: '{url}'. Scheme and domain are required.")

    return {
        "scheme": parsed_url.scheme,
        "domain": parsed_url.netloc,
        "path": parsed_url.path
    }

def build_query_string(params: dict) -> str:
    """
    Builds a URL-encoded query string from a dictionary of parameters.

    Args:
        params: A dictionary of parameters (key-value pairs).

    Returns:
        A URL-encoded query string.
    """
    if not isinstance(params, dict):
        raise ValueError("Input parameters must be a dictionary.")
    return urlencode(params)

def create_api_redirect_url(base_url_api: str, api_endpoint: str, opts: dict) -> str:
    """
    Generates a fully-formed API redirect URL with validation.

    Args:
        base_url_api: The base URL of the API (e.g., "https://api.myservice.com").
        api_endpoint: The API endpoint (e.g., "/v1/data").
        opts: A dictionary of query parameters.

    Returns:
        A complete redirect URL string.

    Raises:
        ValueError: If the base domain is not approved, the endpoint is not allowed,
                    or the base_url_api is malformed.
    """
    if not isinstance(base_url_api, str):
        raise ValueError("base_url_api must be a string.")
    if not isinstance(api_endpoint, str):
        raise ValueError("api_endpoint must be a string.")
    if not isinstance(opts, dict):
        raise ValueError("opts must be a dictionary.")

    parsed_base_url = urlparse(base_url_api)

    if not parsed_base_url.scheme:
        raise ValueError(f"Invalid base_url_api: '{base_url_api}'. Scheme is required (e.g., https).")
    
    if parsed_base_url.netloc not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{parsed_base_url.netloc}' is not an approved API domain.")

    if api_endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{api_endpoint}' is not an allowed API endpoint.")

    query_string = build_query_string(opts)
    
    # Ensure the endpoint starts with a slash if not already, and base_url_api doesn't end with one
    # urljoin handles this gracefully
    full_path = urljoin(parsed_base_url.path + ('/' if parsed_base_url.path and not parsed_base_url.path.endswith('/') else ''), api_endpoint.lstrip('/'))
    
    # Reconstruct the base URL without query or fragment, then add the new path and query
    # Using urljoin to combine base_url_api (scheme+netloc) with the api_endpoint
    # and then adding the query string.
    
    # Construct the path part correctly
    # urljoin(base, path) is good for this
    final_url_path_part = urljoin(base_url_api, api_endpoint)

    # Parse the combined URL to easily add the query string
    parsed_final_url = urlparse(final_url_path_part)
    
    # Create the final URL components
    final_components = (
        parsed_final_url.scheme,
        parsed_final_url.netloc,
        parsed_final_url.path,
        parsed_final_url.params, # usually empty
        query_string,
        parsed_final_url.fragment # usually empty for API calls
    )
    
    return urlunparse(final_components)

if __name__ == '__main__':
    # Example Usage
    test_urls = [
        "http://www.example.com/path/to/resource",
        "https://sub.example.co.uk:8080/another/path?query=123#fragment",
        "ftp://user:pass@example.com/dir/",
        "example.com/path",  # Invalid - no scheme
        "http:///path/only", # Invalid - no domain (netloc can be empty but usually means error for http)
        "htp://example.com/path", # Valid scheme, will be parsed
        "", # Invalid
        None # Invalid type
    ]

    for test_url in test_urls:
        try:
            if test_url is None: # Special case for None to test type check
                 validate_and_parse_url(test_url)
            else:
                components = validate_and_parse_url(test_url)
                print(f"URL: '{test_url}' -> Parsed: {components}")
        except ValueError as e:
            print(f"URL: '{test_url}' -> Error: {e}")

    # Test case for urlparse behavior with missing domain but present scheme
    try:
        components = validate_and_parse_url("http:///path/only")
        print(f"URL: 'http:///path/only' -> Parsed: {components}")
    except ValueError as e:
        print(f"URL: 'http:///path/only' -> Error: {e}")
    
    # Test case for urlparse behavior with missing scheme but present domain
    try:
        components = validate_and_parse_url("www.example.com/path")
        print(f"URL: 'www.example.com/path' -> Parsed: {components}")
    except ValueError as e:
        print(f"URL: 'www.example.com/path' -> Error: {e}")

    # Example Usage for build_query_string
    print("\n--- build_query_string examples ---")
    params1 = {"name": "John Doe", "age": "30", "city": "New York"}
    query_string1 = build_query_string(params1)
    print(f"Parameters: {params1} -> Query String: {query_string1}")

    params2 = {"search": "python programming", "page": "1"}
    query_string2 = build_query_string(params2)
    print(f"Parameters: {params2} -> Query String: {query_string2}")

    params3 = {}
    query_string3 = build_query_string(params3)
    print(f"Parameters: {params3} -> Query String: {query_string3}")

    try:
        build_query_string("not a dict")
    except ValueError as e:
        print(f"Parameters: 'not a dict' -> Error: {e}")

    # Example Usage for create_api_redirect_url
    print("\n--- create_api_redirect_url examples ---")
    
    valid_base = "https://api.myservice.com"
    valid_endpoint = "/v1/data"
    valid_opts = {"id": "123", "format": "json"}

    try:
        redirect_url = create_api_redirect_url(valid_base, valid_endpoint, valid_opts)
        print(f"Base: '{valid_base}', Endpoint: '{valid_endpoint}', Opts: {valid_opts} -> URL: {redirect_url}")
    except ValueError as e:
        print(f"Error creating URL: {e}")

    try:
        # Invalid domain
        redirect_url = create_api_redirect_url("https://api.otherservice.com", valid_endpoint, valid_opts)
        print(f"URL: {redirect_url}")
    except ValueError as e:
        print(f"Error (invalid domain): {e}")

    try:
        # Invalid endpoint
        redirect_url = create_api_redirect_url(valid_base, "/v3/unknown", valid_opts)
        print(f"URL: {redirect_url}")
    except ValueError as e:
        print(f"Error (invalid endpoint): {e}")

    try:
        # Base URL without scheme
        redirect_url = create_api_redirect_url("api.myservice.com", valid_endpoint, valid_opts)
        print(f"URL: {redirect_url}")
    except ValueError as e:
        print(f"Error (base URL no scheme): {e}")

    try:
        # Base URL with path, ensure it's handled
        redirect_url = create_api_redirect_url("https://api.myservice.com/api_base_path", "/v1/user", {"token": "abc"})
        # Expected: https://api.myservice.com/api_base_path/v1/user?token=abc
        print(f"Base with path: 'https://api.myservice.com/api_base_path', Endpoint: '/v1/user' -> URL: {redirect_url}")
    except ValueError as e:
        print(f"Error (base with path): {e}")
    
    try:
        # Health endpoint with no opts
        redirect_url = create_api_redirect_url("https://api-test.myservice.com", "/health", {})
        print(f"Base: 'https://api-test.myservice.com', Endpoint: '/health', Opts: {{}} -> URL: {redirect_url}")
    except ValueError as e:
        print(f"Error (health endpoint): {e}")

    try:
        # Invalid opts type
        redirect_url = create_api_redirect_url(valid_base, valid_endpoint, "not a dict")
        print(f"URL: {redirect_url}")
    except ValueError as e:
        print(f"Error (invalid opts type): {e}")
