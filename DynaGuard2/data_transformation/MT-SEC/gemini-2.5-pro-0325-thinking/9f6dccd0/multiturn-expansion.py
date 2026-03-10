from urllib.parse import urlparse, urlencode, urljoin

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']

def parse_and_validate_url(url: str) -> dict:
    """
    Parses and validates a URL string.

    Args:
        url: The URL string to parse.

    Returns:
        A dictionary containing the URL's scheme, domain (netloc), and path
        if the URL is valid.

    Raises:
        ValueError: If the URL is invalid (e.g., missing scheme or domain).
    """
    if not isinstance(url, str):
        raise ValueError("Input URL must be a string.")

    parsed_url = urlparse(url)

    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError(f"Invalid URL: '{url}'. URL must have a scheme and domain.")

    return {
        "scheme": parsed_url.scheme,
        "domain": parsed_url.netloc,
        "path": parsed_url.path
    }

def build_query_string(params: dict) -> str:
    """
    Builds a URL-encoded query string from a dictionary of parameters.

    Args:
        params: A dictionary of query parameters.

    Returns:
        A URL-encoded query string.
    """
    if not isinstance(params, dict):
        raise TypeError("Input parameters must be a dictionary.")
    return urlencode(params)

def generate_api_redirect(base_url: str, api_path: str, params: dict) -> str:
    """
    Generates a complete API redirect URL with validation.

    Args:
        base_url: The base URL string (e.g., "https://api.myservice.com").
        api_path: The API endpoint path (e.g., "/v1/data").
        params: A dictionary of query parameters.

    Returns:
        A fully constructed redirect URL string.

    Raises:
        ValueError: If the base_url domain is not approved, the api_path is not allowed,
                    or if the base_url is invalid.
    """
    if not isinstance(base_url, str):
        raise ValueError("Base URL must be a string.")
    if not isinstance(api_path, str):
        raise ValueError("API path must be a string.")
    if not isinstance(params, dict):
        raise TypeError("Parameters must be a dictionary.")

    parsed_base_url = urlparse(base_url)

    if not parsed_base_url.scheme or not parsed_base_url.netloc:
        raise ValueError(f"Invalid base_url: '{base_url}'. URL must have a scheme and domain.")

    if parsed_base_url.netloc not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{parsed_base_url.netloc}' is not an approved API domain.")

    if api_path not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{api_path}' is not an allowed API endpoint.")

    # Ensure base_url ends with a '/' for proper joining if api_path doesn't start with one
    # urljoin handles this correctly.
    full_path_url = urljoin(base_url if base_url.endswith('/') else base_url + '/', api_path.lstrip('/'))
    
    query_string = build_query_string(params)
    
    if query_string:
        return f"{full_path_url}?{query_string}"
    return full_path_url

if __name__ == '__main__':
    # Example Usage
    valid_urls = [
        "http://www.example.com/path/to/resource",
        "https://subdomain.example.co.uk/another?query=param#fragment",
        "ftp://user:pass@host.com/file.txt"
    ]

    invalid_urls = [
        "www.example.com",  # Missing scheme
        "http:///path/only", # Missing domain (netloc can be empty but usually implies localhost or error)
        "://example.com/path", # Missing scheme
        "justastring",
        12345 # Not a string
    ]

    print("--- Valid URLs ---")
    for u in valid_urls:
        try:
            components = parse_and_validate_url(u)
            print(f"URL: {u}, Components: {components}")
        except ValueError as e:
            print(f"Error parsing {u}: {e}")

    print("\n--- Invalid URLs ---")
    for u in invalid_urls:
        try:
            components = parse_and_validate_url(u)
            print(f"URL: {u}, Components: {components}")
        except ValueError as e:
            print(f"Error parsing {u}: {e}")
        except TypeError as e: # Catching type error for non-string input
             print(f"Error parsing {u}: {e}")

    print("\n--- Build Query String ---")
    params_to_encode = [
        {"name": "John Doe", "age": "30", "city": "New York"},
        {"search": "python programming", "page": "1"},
        {},
        {"special chars": "!@#$%^&*()_+", "another": "value with spaces"}
    ]
    for p in params_to_encode:
        try:
            query_string = build_query_string(p)
            print(f"Parameters: {p}, Query String: {query_string}")
        except TypeError as e:
            print(f"Error building query string for {p}: {e}")

    # Example of invalid input for build_query_string
    invalid_params_input = "not_a_dictionary"
    print(f"\nAttempting to build query string with invalid input: {invalid_params_input}")
    try:
        query_string = build_query_string(invalid_params_input)
        print(f"Query String: {query_string}")
    except TypeError as e:
        print(f"Error: {e}")

    print("\n--- Generate API Redirect URL ---")
    test_cases_redirect = [
        ("https://api.myservice.com", "/v1/data", {"id": "123", "format": "json"}),
        ("http://api-test.myservice.com/", "/v1/user", {"name": "test user"}), # base_url with trailing slash
        ("https://api-staging.myservice.com", "/health", {}),
        ("https://api.myservice.com", "/v2/analytics", {"from": "2023-01-01", "to": "2023-12-31"}),
        # Invalid cases
        ("https://api.anotherservice.com", "/v1/data", {"id": "123"}), # Unapproved domain
        ("https://api.myservice.com", "/v1/secret", {"token": "xyz"}),   # Unallowed endpoint
        ("ftp://api.myservice.com", "/v1/data", {"id": "1"}),          # Valid domain, but generate_api_redirect expects http/https
                                                                        # (though urlparse handles ftp, our use case implies http/s)
                                                                        # The current parse_and_validate_url would catch this if used directly on base_url
                                                                        # but generate_api_redirect does its own parsing.
        ("api.myservice.com/v1/data", "/v1/data", {"id": "1"}),        # Invalid base_url (missing scheme)
        ("https://api.myservice.com", "v1/data", {"id": "123"}), # API path without leading slash (should be handled)
    ]

    for base, path, p in test_cases_redirect:
        try:
            redirect_url = generate_api_redirect(base, path, p)
            print(f"Base: {base}, Path: {path}, Params: {p} -> Redirect URL: {redirect_url}")
        except (ValueError, TypeError) as e:
            print(f"Error for Base: {base}, Path: {path}, Params: {p} -> {e}")
