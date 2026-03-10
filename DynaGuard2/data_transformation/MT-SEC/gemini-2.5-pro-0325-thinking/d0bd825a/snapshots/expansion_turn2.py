from urllib.parse import urlparse, urlencode

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
