from urllib.parse import urlparse, urlencode, urlunparse

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
        raise ValueError(f"Invalid URL: '{url}'. URL must have a scheme and netloc.")
    
    return {
        "scheme": parsed_url.scheme,
        "netloc": parsed_url.netloc,
        "path": parsed_url.path,
    }

def generate_query_string(params: dict) -> str:
    """
    Generates a URL-encoded query string from a dictionary of parameters.

    Args:
        params: A dictionary of parameters (keys and values).

    Returns:
        A URL-encoded query string.
    """
    return urlencode(params)

def create_redirect_url(base_url_api: str, desired_endpoint: str, optional_params: dict = None) -> str:
    """
    Creates a complete API redirect URL with validation.

    Args:
        base_url_api: The base URL of the API (e.g., "https://api.myservice.com").
        desired_endpoint: The desired API endpoint (e.g., "/v1/user").
        optional_params: A dictionary of optional query parameters.

    Returns:
        A complete redirect URL string.

    Raises:
        ValueError: If the base domain is not approved, the endpoint is not allowed,
                    or the base_url_api is invalid.
    """
    parsed_base_url = urlparse(base_url_api)

    if not parsed_base_url.scheme or not parsed_base_url.netloc:
        raise ValueError(f"Invalid base_url_api: '{base_url_api}'. URL must have a scheme and netloc.")

    if parsed_base_url.netloc not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{parsed_base_url.netloc}' is not an approved API domain.")

    if desired_endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{desired_endpoint}' is not an allowed endpoint.")

    query_string = ""
    if optional_params:
        query_string = urlencode(optional_params)

    # Reconstruct the URL ensuring no double slashes between netloc and path,
    # and path and endpoint. The endpoint should start with a '/'.
    # urlunparse components: (scheme, netloc, path, params, query, fragment)
    # We combine the path from base_url (if any) with the desired_endpoint.
    # Ensure desired_endpoint starts with '/' and base_path (if exists) doesn't end with '/'
    # or vice-versa to avoid double slashes.
    
    final_path = parsed_base_url.path.rstrip('/') + desired_endpoint

    redirect_url_parts = (
        parsed_base_url.scheme,
        parsed_base_url.netloc,
        final_path,
        '',  # params for path segments, not query string
        query_string,
        ''   # fragment
    )
    return urlunparse(redirect_url_parts)

if __name__ == '__main__':
    # Example Usage
    test_urls = [
        "http://www.example.com/path/to/resource",
        "https://subdomain.example.co.uk:8080/another/path?query=param#fragment",
        "ftp://user:pass@example.com/dir/",
        "invalid-url",
        "www.missing-scheme.com",
        "http:///missing-netloc",
    ]

    for u in test_urls:
        try:
            components = parse_and_validate_url(u)
            print(f"Parsed '{u}': {components}")
        except ValueError as e:
            print(f"Error parsing '{u}': {e}")

    # Example of accessing components
    try:
        valid_url = "https://docs.python.org/3/library/urllib.parse.html"
        parsed = parse_and_validate_url(valid_url)
        print(f"\nScheme of '{valid_url}': {parsed['scheme']}")
        print(f"Netloc of '{valid_url}': {parsed['netloc']}")
        print(f"Path of '{valid_url}': {parsed['path']}")
    except ValueError as e:
        print(f"Error: {e}")

    # Example of generating query string
    params_dict = {"name": "John Doe", "age": "30", "city": "New York"}
    query_string = generate_query_string(params_dict)
    print(f"\nGenerated query string for {params_dict}: {query_string}")

    params_with_special_chars = {"search": "python programming", "page": "1"}
    query_string_special = generate_query_string(params_with_special_chars)
    print(f"Generated query string for {params_with_special_chars}: {query_string_special}")

    empty_params = {}
    query_string_empty = generate_query_string(empty_params)
    print(f"Generated query string for {empty_params}: '{query_string_empty}'")

    # Example of creating redirect URL
    print("\n--- Create Redirect URL Examples ---")
    test_redirects = [
        ("https://api.myservice.com", "/v1/user", {"id": "123", "format": "json"}),
        ("http://api-test.myservice.com/api_base_path", "/v1/data", {"filter": "active"}),
        ("https://api-staging.myservice.com", "/health", None),
        ("https://api.anotherservice.com", "/v1/user", {"id": "123"}), # Invalid domain
        ("https://api.myservice.com", "/v1/unknown", {"id": "123"}),   # Invalid endpoint
        ("invalid-base-url", "/v1/user", {}),                         # Invalid base URL
        ("https://api.myservice.com", "/v1/user", {"name": "test user", "status": "active&pending"}), # Params to be encoded
    ]

    for base, endpoint, params in test_redirects:
        try:
            redirect_url = create_redirect_url(base, endpoint, params)
            print(f"Redirect URL for ('{base}', '{endpoint}', {params}): {redirect_url}")
        except ValueError as e:
            print(f"Error for ('{base}', '{endpoint}', {params}): {e}")
