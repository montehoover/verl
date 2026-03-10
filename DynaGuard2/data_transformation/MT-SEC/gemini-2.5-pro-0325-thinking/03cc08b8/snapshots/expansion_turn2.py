from urllib.parse import urlparse, urlencode, urljoin

ALLOWED_DOMAINS = [
    "example.com",
    "trusted.org",
    "sub.example.com",
]

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
