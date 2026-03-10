from urllib.parse import urlparse, urlencode

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
