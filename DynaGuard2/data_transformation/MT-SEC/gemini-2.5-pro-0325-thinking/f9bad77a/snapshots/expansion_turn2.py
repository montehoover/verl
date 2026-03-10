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
        raise ValueError(f"Invalid URL: {url}. Scheme and netloc are required.")
    
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

if __name__ == '__main__':
    # Example Usage
    test_urls = [
        "http://www.example.com/path/to/resource",
        "https://subdomain.example.co.uk:8080/another/path?query=string#fragment",
        "ftp://user:password@example.com/resource",
        "invalid-url",
        "www.missing-scheme.com",
        "http:///missing-netloc",
    ]

    for url_str in test_urls:
        try:
            components = parse_and_validate_url(url_str)
            print(f"Parsed '{url_str}': {components}")
        except ValueError as e:
            print(f"Error parsing '{url_str}': {e}")

    # Example of a valid URL
    try:
        valid_url = "https://docs.python.org/3/library/urllib.parse.html"
        components = parse_and_validate_url(valid_url)
        print(f"\nSuccessfully parsed valid URL '{valid_url}': {components}")
    except ValueError as e:
        print(f"\nError with valid URL '{valid_url}': {e}")

    # Example of an invalid URL (missing scheme)
    try:
        invalid_url_no_scheme = "example.com/path"
        components = parse_and_validate_url(invalid_url_no_scheme)
        print(f"\nSuccessfully parsed invalid URL '{invalid_url_no_scheme}': {components}")
    except ValueError as e:
        print(f"\nError with invalid URL (no scheme) '{invalid_url_no_scheme}': {e}")
    
    # Example of an invalid URL (missing netloc, though urlparse might still parse it)
    # According to the requirement, scheme AND netloc must be present.
    try:
        invalid_url_no_netloc = "http:///path" 
        components = parse_and_validate_url(invalid_url_no_netloc)
        print(f"\nSuccessfully parsed invalid URL '{invalid_url_no_netloc}': {components}")
    except ValueError as e:
        print(f"\nError with invalid URL (no netloc) '{invalid_url_no_netloc}': {e}")

    # Example Usage for generate_query_string
    print("\n--- generate_query_string examples ---")
    params1 = {"name": "John Doe", "age": "30", "city": "New York"}
    query_string1 = generate_query_string(params1)
    print(f"Parameters: {params1}, Query String: {query_string1}")

    params2 = {"search": "python programming", "page": "1", "sort_by": "relevance"}
    query_string2 = generate_query_string(params2)
    print(f"Parameters: {params2}, Query String: {query_string2}")

    params3 = {"email": "test@example.com", "subscribe": "true"}
    query_string3 = generate_query_string(params3)
    print(f"Parameters: {params3}, Query String: {query_string3}")

    params4 = {} # Empty parameters
    query_string4 = generate_query_string(params4)
    print(f"Parameters: {params4}, Query String: {query_string4}")

    params5 = {"special_chars": "!@#$%^&*()_+", "another": "value with spaces"}
    query_string5 = generate_query_string(params5)
    print(f"Parameters: {params5}, Query String: {query_string5}")
