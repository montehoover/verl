from urllib.parse import urlparse, urlencode

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
