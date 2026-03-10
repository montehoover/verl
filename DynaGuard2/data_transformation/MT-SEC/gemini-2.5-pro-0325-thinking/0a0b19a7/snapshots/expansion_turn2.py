from urllib.parse import urlparse, urlencode

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
