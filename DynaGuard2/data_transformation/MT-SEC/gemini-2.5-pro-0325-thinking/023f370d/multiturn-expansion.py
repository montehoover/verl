from urllib.parse import urlparse, urlencode, urlunparse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']

def validate_and_parse_url(url: str) -> dict:
    """
    Validates and parses a URL string.

    Args:
        url: The URL string to validate and parse.

    Returns:
        A dictionary with 'scheme', 'domain', and 'path' if the URL is valid.

    Raises:
        ValueError: If the URL is invalid (e.g., missing scheme or domain).
    """
    if not isinstance(url, str):
        raise ValueError("Input URL must be a string.")

    parsed_url = urlparse(url)

    if not parsed_url.scheme:
        raise ValueError(f"Invalid URL '{url}': Missing scheme (e.g., http, https).")
    
    if not parsed_url.netloc:
        raise ValueError(f"Invalid URL '{url}': Missing domain name.")

    return {
        'scheme': parsed_url.scheme,
        'domain': parsed_url.netloc,
        'path': parsed_url.path
    }

def build_query_string(params: dict) -> str:
    """
    Builds a URL-encoded query string from a dictionary of parameters.

    Args:
        params: A dictionary of query parameters (key-value pairs).

    Returns:
        A URL-encoded query string.
    """
    if not isinstance(params, dict):
        raise ValueError("Input parameters must be a dictionary.")
    
    return urlencode(params)

def construct_api_redirect(base_api_url: str, api_endpoint: str, query_options: dict) -> str:
    """
    Constructs a complete API redirect URL with validation.

    Args:
        base_api_url: The base URL for the API (e.g., "https://api.myservice.com").
        api_endpoint: The specific API endpoint (e.g., "/v1/data").
        query_options: A dictionary of query parameters.

    Returns:
        A fully constructed redirect URL string.

    Raises:
        ValueError: If the base domain is not approved, the endpoint is not allowed,
                    or the base_api_url is invalid.
    """
    parsed_base_url = urlparse(base_api_url)

    if not parsed_base_url.scheme or not parsed_base_url.netloc:
        raise ValueError(f"Invalid base_api_url: '{base_api_url}'. Must include scheme and domain.")

    if parsed_base_url.netloc not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{parsed_base_url.netloc}' is not an approved API domain.")

    if api_endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{api_endpoint}' is not an allowed API endpoint.")

    # Ensure endpoint starts with a slash if not empty, and base_api_url doesn't end with one
    # to prevent double slashes. urlparse components handle this well, but good to be mindful.
    # urlunparse will correctly join path segments.

    query_string = build_query_string(query_options)
    
    # Construct the final URL.
    # scheme, netloc, path, params, query, fragment
    # We are building path from base_api_url.path + api_endpoint
    # Ensure path joining is correct (e.g. base might have a path prefix)
    
    final_path = parsed_base_url.path.rstrip('/') + '/' + api_endpoint.lstrip('/')
    if not api_endpoint.startswith('/'): # handle cases where endpoint might not have leading /
        final_path = parsed_base_url.path.rstrip('/') + '/' + api_endpoint
    else:
        final_path = parsed_base_url.path.rstrip('/') + api_endpoint
    
    # Normalize path to remove any double slashes that might have been introduced
    # if base_api_url.path was '/' and api_endpoint also started with '/'
    # However, urlparse and urlunparse usually handle this.
    # A more robust way for path joining:
    path_parts = [part for part in parsed_base_url.path.split('/') if part]
    endpoint_parts = [part for part in api_endpoint.split('/') if part]
    combined_path = "/" + "/".join(path_parts + endpoint_parts)


    redirect_url_parts = (
        parsed_base_url.scheme,
        parsed_base_url.netloc,
        combined_path,
        '',  # params (matrix parameters, not query string)
        query_string,
        ''   # fragment
    )
    
    return urlunparse(redirect_url_parts)

if __name__ == '__main__':
    # Example Usage
    test_urls = [
        "https://www.example.com/path/to/resource",
        "http://localhost:8080/api/v1/users",
        "ftp://files.example.com/uploads/file.txt",
        "www.example.com/path",  # Invalid: missing scheme
        "https://",  # Invalid: missing domain
        "https://example.com", # Valid
        "http://example", # Valid (though 'example' might not resolve, it's structurally valid)
        12345 # Invalid type
    ]

    for test_url in test_urls:
        try:
            print(f"Processing URL: {test_url}")
            result = validate_and_parse_url(test_url)
            print(f"Parsed URL: {result}")
        except ValueError as e:
            print(f"Error: {e}")
        print("-" * 20)

    # Example Usage for construct_api_redirect
    print("\nTesting construct_api_redirect:")
    redirect_tests = [
        ("https://api.myservice.com", "/v1/data", {"id": "123", "format": "json"}),
        ("http://api-test.myservice.com/api_prefix", "/v1/user", {"user_id": "456"}), # base with path
        ("https://api-staging.myservice.com", "/health", {}),
        ("https://api.anotherservice.com", "/v1/data", {"key": "value"}),  # Invalid domain
        ("https://api.myservice.com", "/v1/admin", {"action": "delete"}),  # Invalid endpoint
        ("api.myservice.com/v1/data", "/v1/data", {}), # Invalid base_api_url (no scheme)
        ("https://api.myservice.com", "v1/data", {"id": "789"}), # Endpoint without leading slash (should be handled)
    ]

    for base_url, endpoint, options in redirect_tests:
        try:
            redirect_url = construct_api_redirect(base_url, endpoint, options)
            print(f"Base: {base_url}, Endpoint: {endpoint}, Options: {options}")
            print(f"Redirect URL: {redirect_url}")
        except ValueError as e:
            print(f"Error processing ({base_url}, {endpoint}, {options}): {e}")
        print("-" * 20)

    # Example Usage for build_query_string
    print("\nTesting build_query_string:")
    param_dicts = [
        {"name": "John Doe", "age": "30", "city": "New York"},
        {"search": "python programming", "page": "1"},
        {}, # Empty dictionary
        {"key with spaces": "value with spaces", "special_chars": "!@#$%^&*()"}
    ]

    for params in param_dicts:
        try:
            query_string = build_query_string(params)
            print(f"Parameters: {params}")
            print(f"Query String: {query_string}")
        except ValueError as e:
            print(f"Error: {e}")
        print("-" * 20)
