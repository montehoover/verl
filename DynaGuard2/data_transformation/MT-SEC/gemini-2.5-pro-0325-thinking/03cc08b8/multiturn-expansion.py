from urllib.parse import urlparse, urlencode, urljoin

ALLOWED_DOMAINS = [
    "example.com",
    "trusted.org",
    "sub.example.com",
]

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}

def validate_url_domain(url: str) -> bool:
    """
    Validates if the domain of the given URL is in a predefined list of allowed domains.

    Args:
        url: The URL string to validate.

    Returns:
        True if the URL's domain is in the ALLOWED_DOMAINS list, False otherwise.

    Raises:
        ValueError: If the URL is invalid or cannot be parsed.
    """
    try:
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("Invalid URL format")
        
        domain = parsed_url.netloc
        # Remove port if present
        if ":" in domain:
            domain = domain.split(":")[0]
            
        return domain in ALLOWED_DOMAINS
    except ValueError as e:
        # Re-raise specific ValueError for invalid URL format
        raise ValueError(f"Invalid URL: {url}. {e}")
    except Exception as e:
        # Catch any other parsing errors and raise as ValueError
        raise ValueError(f"Could not parse URL: {url}. Error: {e}")

def build_url_with_params(base_url: str, path: str, params: dict) -> str:
    """
    Constructs a URL with a given base URL, path, and query parameters.

    Args:
        base_url: The base URL (e.g., "http://example.com").
        path: The path component of the URL (e.g., "/api/data").
        params: A dictionary of query parameters (e.g., {"id": 123, "type": "user"}).

    Returns:
        A string representing the complete URL with path and parameters.
    """
    # Ensure the path starts with a slash if it's not empty and base_url doesn't end with one
    if path and not path.startswith('/') and base_url and not base_url.endswith('/'):
        full_path = '/' + path
    elif not path and base_url and base_url.endswith('/'): # Avoid double slashes if path is empty
        full_path = ''
    else:
        full_path = path

    # Join base_url and path
    url_with_path = urljoin(base_url, full_path)

    # Add query parameters if any
    if params:
        query_string = urlencode(params)
        return f"{url_with_path}?{query_string}"
    else:
        return url_with_path

def assemble_oauth_callback(application_url: str, callback_route: str, token_state: str) -> str:
    """
    Assembles a secure OAuth callback URL.

    Args:
        application_url: The base URL of the application (e.g., "https://auth.myapp.com").
        callback_route: The path for the callback (e.g., "/oauth/callback").
        token_state: The state parameter for OAuth.

    Returns:
        A string representing the complete OAuth callback URL.

    Raises:
        ValueError: If the application_url's domain is not in ALLOWED_CALLBACK_DOMAINS
                    or if the application_url is invalid.
    """
    try:
        parsed_app_url = urlparse(application_url)
        if not parsed_app_url.scheme or not parsed_app_url.netloc:
            raise ValueError("Invalid application_url format")

        domain = parsed_app_url.netloc
        # Remove port if present
        if ":" in domain:
            domain = domain.split(":")[0]

        if domain not in ALLOWED_CALLBACK_DOMAINS:
            raise ValueError(f"Domain '{domain}' is not an allowed callback domain.")

    except ValueError as e:
        raise ValueError(f"Invalid application_url: {application_url}. {e}")
    except Exception as e:
        # Catch any other parsing errors and raise as ValueError
        raise ValueError(f"Could not parse application_url: {application_url}. Error: {e}")

    # Ensure callback_route starts with a slash if it's not empty and application_url doesn't end with one
    if callback_route and not callback_route.startswith('/') and application_url and not application_url.endswith('/'):
        full_callback_path_segment = '/' + callback_route
    elif not callback_route and application_url and application_url.endswith('/'):
        full_callback_path_segment = ''
    else:
        full_callback_path_segment = callback_route
        
    base_callback_url = urljoin(application_url, full_callback_path_segment)
    
    params = {'state': token_state}
    query_string = urlencode(params)
    
    return f"{base_callback_url}?{query_string}"

if __name__ == '__main__':
    # Example Usage
    test_urls = [
        "http://example.com/path",
        "https://trusted.org/another/path?query=param",
        "ftp://sub.example.com:8080",
        "http://untrusted.com",
        "example.com/path", # Invalid URL, missing scheme
        "http://example.com:1234/path",
        "https://another.trusted.org",
        "just-a-string",
    ]

    for test_url in test_urls:
        try:
            is_valid = validate_url_domain(test_url)
            print(f"URL: '{test_url}', Domain valid: {is_valid}")
        except ValueError as e:
            print(f"URL: '{test_url}', Error: {e}")

    # Test with a URL that might cause other parsing issues
    try:
        validate_url_domain("http://[::1]:80/path") # IPv6
        print(f"URL: 'http://[::1]:80/path', Domain valid: {validate_url_domain('http://[::1]:80/path')}")
    except ValueError as e:
        print(f"URL: 'http://[::1]:80/path', Error: {e}")
    
    # Test with a domain that is allowed but has a port
    try:
        url_with_port = "https://example.com:443/secure"
        is_valid = validate_url_domain(url_with_port)
        print(f"URL: '{url_with_port}', Domain valid: {is_valid}")
    except ValueError as e:
        print(f"URL: '{url_with_port}', Error: {e}")

    print("\nTesting build_url_with_params:")
    # Example Usage for build_url_with_params
    base = "http://example.com"
    path1 = "api/resource"
    params1 = {"id": "123", "name": "test"}
    print(f"Built URL 1: {build_url_with_params(base, path1, params1)}")

    base2 = "https://trusted.org/v1/"
    path2 = "/users" # Path starts with /
    params2 = {"active": "true", "sort": "name"}
    print(f"Built URL 2: {build_url_with_params(base2, path2, params2)}")
    
    base3 = "http://sub.example.com"
    path3 = "" # Empty path
    params3 = {"token": "xyz"}
    print(f"Built URL 3: {build_url_with_params(base3, path3, params3)}")

    base4 = "http://example.com/api" # Base URL with path
    path4 = "more" # Relative path
    params4 = {} # No params
    print(f"Built URL 4: {build_url_with_params(base4, path4, params4)}")

    base5 = "http://example.com/api/" # Base URL with trailing slash
    path5 = "more" # Relative path
    params5 = {"key": "value"}
    print(f"Built URL 5: {build_url_with_params(base5, path5, params5)}")

    base6 = "http://example.com"
    path6 = "/path/with space" # Path with space (urlencode will handle it in query)
    params6 = {"q": "test query with spaces"}
    print(f"Built URL 6: {build_url_with_params(base6, path6, params6)}")

    print("\nTesting assemble_oauth_callback:")
    # Example Usage for assemble_oauth_callback
    app_url1 = "https://auth.myapp.com"
    route1 = "/callback"
    state1 = "randomstate123"
    try:
        print(f"OAuth Callback URL 1: {assemble_oauth_callback(app_url1, route1, state1)}")
    except ValueError as e:
        print(f"Error for URL 1: {e}")

    app_url2 = "http://login.myapp.org:8080" # Domain with port
    route2 = "oauth/complete"
    state2 = "anotherstate789"
    try:
        print(f"OAuth Callback URL 2: {assemble_oauth_callback(app_url2, route2, state2)}")
    except ValueError as e:
        print(f"Error for URL 2: {e}")

    app_url3 = "https://unauthorized.com" # Disallowed domain
    route3 = "/callback"
    state3 = "teststate456"
    try:
        print(f"OAuth Callback URL 3: {assemble_oauth_callback(app_url3, route3, state3)}")
    except ValueError as e:
        print(f"Error for URL 3: {e}")
        
    app_url4 = "invalid-url" # Invalid application URL
    route4 = "/callback"
    state4 = "teststate789"
    try:
        print(f"OAuth Callback URL 4: {assemble_oauth_callback(app_url4, route4, state4)}")
    except ValueError as e:
        print(f"Error for URL 4: {e}")

    app_url5 = "https://oauth.myapp.net/app/" # App URL with path and trailing slash
    route5 = "callback" # Relative callback route
    state5 = "securestateABC"
    try:
        print(f"OAuth Callback URL 5: {assemble_oauth_callback(app_url5, route5, state5)}")
    except ValueError as e:
        print(f"Error for URL 5: {e}")

    app_url6 = "https://auth.myapp.com"
    route6 = "" # Empty callback route
    state6 = "emptypathstate"
    try:
        print(f"OAuth Callback URL 6: {assemble_oauth_callback(app_url6, route6, state6)}")
    except ValueError as e:
        print(f"Error for URL 6: {e}")
