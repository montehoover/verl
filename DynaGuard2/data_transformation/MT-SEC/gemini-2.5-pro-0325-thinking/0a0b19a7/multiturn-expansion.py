from urllib.parse import urlparse, urlencode, urljoin

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
        ValueError: If the URL is invalid (e.g., missing scheme or netloc).
    """
    parsed_url = urlparse(url)
    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError(f"Invalid URL: '{url}'. URL must have a scheme and a network location.")
    
    return {
        "scheme": parsed_url.scheme,
        "netloc": parsed_url.netloc,
        "path": parsed_url.path,
    }

def generate_query_string(params: dict) -> str:
    """
    Generates a URL-encoded query string from a dictionary of parameters.

    Args:
        params: A dictionary of query parameters (key-value pairs).

    Returns:
        A URL-encoded query string.
    """
    return urlencode(params)

def build_api_redirect_url(api_base_url: str, endpoint: str, query_params: dict) -> str:
    """
    Constructs a complete API redirect URL with validation.

    Args:
        api_base_url: The base URL of the API (e.g., "https://api.myservice.com").
        endpoint: The API endpoint (e.g., "/v1/data").
        query_params: A dictionary of query parameters.

    Returns:
        A complete redirect URL string.

    Raises:
        ValueError: If the base domain is not approved, the endpoint is not allowed,
                    or the base URL is invalid.
    """
    parsed_base_url = urlparse(api_base_url)
    if not parsed_base_url.scheme or not parsed_base_url.netloc:
        raise ValueError(f"Invalid api_base_url: '{api_base_url}'. URL must have a scheme and a network location.")

    if parsed_base_url.netloc not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{parsed_base_url.netloc}' is not an approved API domain.")

    if endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{endpoint}' is not an allowed API endpoint.")

    # Ensure the base URL ends with a slash for proper joining, if it doesn't have a path
    # or its path doesn't end with a slash.
    # urljoin handles this well, but being explicit can sometimes clarify intent.
    # For example, urljoin("http://host.com", "/path") is "http://host.com/path"
    # urljoin("http://host.com/api", "/path") is "http://host.com/path" (undesired)
    # urljoin("http://host.com/api/", "path") is "http://host.com/api/path" (desired)
    # urljoin("http://host.com/api/", "/path") is "http://host.com/path" (undesired if endpoint is relative)
    # Since our endpoints start with '/', urljoin should work correctly.

    full_path_url = urljoin(api_base_url, endpoint)
    
    query_string = generate_query_string(query_params)
    if query_string:
        return f"{full_path_url}?{query_string}"
    return full_path_url

if __name__ == '__main__':
    # Example Usage
    test_urls = [
        "http://www.example.com/path/to/resource",
        "https://subdomain.example.co.uk:8080/another/path?query=string#fragment",
        "ftp://user:password@example.com/resource.txt",
        "invalid-url",
        "www.missing-scheme.com",
        "http:///missing-netloc",
    ]

    for test_url in test_urls:
        try:
            components = parse_and_validate_url(test_url)
            print(f"Parsed '{test_url}': {components}")
        except ValueError as e:
            print(f"Error parsing '{test_url}': {e}")

    # Example of accessing components
    try:
        valid_url = "https://docs.python.org/3/library/urllib.parse.html"
        parsed = parse_and_validate_url(valid_url)
        print(f"\nScheme of '{valid_url}': {parsed['scheme']}")
        print(f"Netloc of '{valid_url}': {parsed['netloc']}")
        print(f"Path of '{valid_url}': {parsed['path']}")
    except ValueError as e:
        print(f"Error: {e}")

    # Example of generating a query string
    query_params = {"name": "John Doe", "age": "30", "city": "New York"}
    query_string = generate_query_string(query_params)
    print(f"\nGenerated query string for {query_params}: {query_string}")

    empty_params = {}
    query_string_empty = generate_query_string(empty_params)
    print(f"Generated query string for {empty_params}: {query_string_empty}")

    params_with_special_chars = {"search": "python programming", "topic": "url encoding & decoding"}
    query_string_special = generate_query_string(params_with_special_chars)
    print(f"Generated query string for {params_with_special_chars}: {query_string_special}")

    # Example of building API redirect URL
    print("\n--- Building API Redirect URLs ---")
    test_api_builds = [
        ("https://api.myservice.com", "/v1/data", {"id": "123", "format": "json"}),
        ("http://api-test.myservice.com/api/", "/v1/user", {"user_id": "456"}), # Test base url with path
        ("https://api-staging.myservice.com", "/health", {}),
        ("https://api.anotherservice.com", "/v1/data", {"id": "789"}), # Invalid domain
        ("https://api.myservice.com", "/v1/unknown", {"key": "value"}),   # Invalid endpoint
        ("api.myservice.com", "/v1/data", {"id": "000"}), # Invalid base URL (missing scheme)
        ("https://api.myservice.com", "v1/data", {"id": "111"}), # Invalid endpoint (missing leading /)
    ]

    for base, endpoint, params in test_api_builds:
        try:
            redirect_url = build_api_redirect_url(base, endpoint, params)
            print(f"Built URL for ('{base}', '{endpoint}', {params}): {redirect_url}")
        except ValueError as e:
            print(f"Error building URL for ('{base}', '{endpoint}', {params}): {e}")
