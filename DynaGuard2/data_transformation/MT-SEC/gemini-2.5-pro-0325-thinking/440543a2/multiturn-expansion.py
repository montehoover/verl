from urllib.parse import urlparse, urlencode, urlunparse, urljoin

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']

def parse_and_validate_url(url: str) -> dict:
    """
    Parses and validates a URL string.

    Args:
        url: The URL string to parse.

    Returns:
        A dictionary containing the URL's scheme, netloc, and path.

    Raises:
        ValueError: If the URL is invalid or missing essential components.
    """
    if not isinstance(url, str):
        raise ValueError("Input URL must be a string.")

    parsed_url = urlparse(url)

    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError(f"Invalid URL: '{url}'. Scheme and netloc are required.")

    return {
        "scheme": parsed_url.scheme,
        "netloc": parsed_url.netloc,
        "path": parsed_url.path,
    }

def build_query_string(params: dict) -> str:
    """
    Builds a URL-encoded query string from a dictionary of parameters.

    Args:
        params: A dictionary of parameters.

    Returns:
        A URL-encoded query string.
    """
    if not isinstance(params, dict):
        raise ValueError("Input parameters must be a dictionary.")
    return urlencode(params)

def create_api_redirect_link(base_url: str, path: str, params: dict) -> str:
    """
    Generates a complete API redirect URL with validation.

    Args:
        base_url: The base URL string (e.g., "https://api.myservice.com").
        path: The API endpoint path (e.g., "/v1/data").
        params: A dictionary of query parameters.

    Returns:
        A fully constructed and validated API URL string.

    Raises:
        ValueError: If the base_url domain is not approved, the path is not allowed,
                    or if inputs are invalid.
    """
    if not isinstance(base_url, str):
        raise ValueError("Base URL must be a string.")
    if not isinstance(path, str):
        raise ValueError("Path must be a string.")
    if not isinstance(params, dict):
        raise ValueError("Params must be a dictionary.")

    parsed_base_url = urlparse(base_url)

    if not parsed_base_url.scheme or not parsed_base_url.netloc:
        raise ValueError(f"Invalid base_url: '{base_url}'. Scheme and netloc are required.")

    if parsed_base_url.netloc not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{parsed_base_url.netloc}' is not an approved API domain.")

    if path not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{path}' is not an allowed API endpoint.")

    # Ensure path starts with a slash if not already, and base_url doesn't end with one if path starts with one.
    # urljoin handles this gracefully.
    full_path_url = urljoin(base_url if base_url.endswith('/') else base_url + '/', path.lstrip('/'))

    query_string = build_query_string(params)
    
    final_url_parts = list(urlparse(full_path_url))
    final_url_parts[4] = query_string # Index 4 is for query string
    
    return urlunparse(final_url_parts)

if __name__ == '__main__':
    # Example Usage
    test_urls = [
        "http://www.example.com/path/to/resource",
        "https://subdomain.example.co.uk:8080/another/path?query=string#fragment",
        "ftp://user:password@host.com/resource",
        "invalid-url",
        "www.missing-scheme.com",
        "http:///missing-netloc",
        12345  # Invalid type
    ]

    for test_url in test_urls:
        try:
            result = parse_and_validate_url(test_url)
            print(f"Parsed '{test_url}': {result}")
        except ValueError as e:
            print(f"Error parsing '{test_url}': {e}")

    print("\nTesting with a valid URL:")
    try:
        valid_url = "https://docs.python.org/3/library/urllib.parse.html"
        components = parse_and_validate_url(valid_url)
        print(f"Scheme: {components['scheme']}")
        print(f"Netloc: {components['netloc']}")
        print(f"Path: {components['path']}")
    except ValueError as e:
        print(f"Error: {e}")

    print("\nTesting create_api_redirect_link:")
    try:
        valid_api_url = create_api_redirect_link(
            "https://api.myservice.com",
            "/v1/data",
            {"id": "123", "format": "json"}
        )
        print(f"Valid API URL: {valid_api_url}")

        valid_api_url_staging = create_api_redirect_link(
            "https://api-staging.myservice.com/", # with trailing slash
            "/v2/analytics",
            {"user": "test_user", "event": "login"}
        )
        print(f"Valid Staging API URL: {valid_api_url_staging}")

        print("\nTesting invalid domain:")
        create_api_redirect_link(
            "https://api.otherservice.com",
            "/v1/data",
            {"id": "456"}
        )
    except ValueError as e:
        print(f"Error: {e}")

    print("\nTesting not allowed endpoint:")
    try:
        create_api_redirect_link(
            "https://api-test.myservice.com",
            "/v3/unknown",
            {"key": "value"}
        )
    except ValueError as e:
        print(f"Error: {e}")

    print("\nTesting invalid base_url format:")
    try:
        create_api_redirect_link(
            "api.myservice.com", # missing scheme
            "/health",
            {}
        )
    except ValueError as e:
        print(f"Error: {e}")
    
    print("\nTesting with empty params:")
    try:
        health_check_url = create_api_redirect_link(
            "http://api.myservice.com", # http is also fine if domain is approved
            "/health",
            {}
        )
        print(f"Health Check URL: {health_check_url}")
    except ValueError as e:
        print(f"Error: {e}")

    print("\nTesting build_query_string:")
    params_dict = {"name": "John Doe", "age": "30", "city": "New York"}
    query_string = build_query_string(params_dict)
    print(f"Parameters: {params_dict}, Query String: '{query_string}'")

    params_with_special_chars = {"key": "value with spaces & symbols", "another_key": "test@example.com"}
    query_string_special = build_query_string(params_with_special_chars)
    print(f"Parameters: {params_with_special_chars}, Query String: '{query_string_special}'")

    empty_params = {}
    query_string_empty = build_query_string(empty_params)
    print(f"Parameters: {empty_params}, Query String: '{query_string_empty}'")

    try:
        build_query_string("not a dict")
    except ValueError as e:
        print(f"Error with invalid input type: {e}")

    print("\nTesting with an invalid URL (missing scheme):")
    try:
        invalid_url = "example.com/path"
        parse_and_validate_url(invalid_url)
    except ValueError as e:
        print(f"Error: {e}")

    print("\nTesting with an invalid URL (missing netloc):")
    try:
        invalid_url = "http://"
        parse_and_validate_url(invalid_url)
    except ValueError as e:
        print(f"Error: {e}")
