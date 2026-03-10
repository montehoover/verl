from urllib.parse import urlparse, urlencode, urljoin

def parse_and_validate_url(url: str) -> tuple[str, str, str]:
    """
    Parses and validates a URL.

    Args:
        url: The URL string to parse.

    Returns:
        A tuple containing the scheme, netloc, and path of the URL.

    Raises:
        ValueError: If the URL is invalid (e.g., missing scheme or netloc).
    """
    parsed_url = urlparse(url)
    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError(f"Invalid URL: '{url}'. URL must include a scheme and netloc.")
    return parsed_url.scheme, parsed_url.netloc, parsed_url.path


def build_url_with_query(base: str, path: str, params: dict) -> str:
    """
    Constructs a full URL with a query string from a base URL, path, and parameters.

    Args:
        base: The base URL (e.g., "http://www.example.com").
        path: The path component of the URL (e.g., "/search").
        params: A dictionary of query parameters (e.g., {"q": "python", "lang": "en"}).

    Returns:
        A string representing the full URL with the query string.
    """
    # Ensure the base URL ends with a slash and the path doesn't start with one,
    # or use urljoin to handle this robustly.
    base_url_with_path = urljoin(base if base.endswith('/') else base + '/', path.lstrip('/'))
    
    if not params:
        return base_url_with_path
        
    query_string = urlencode(params)
    return f"{base_url_with_path}?{query_string}"

if __name__ == '__main__':
    # Example Usage
    test_urls = [
        "http://www.example.com/path/to/page",
        "https://subdomain.example.co.uk:8080/another/path?query=string#fragment",
        "ftp://user:password@example.com/resource",
        "www.example.com/missing/scheme",  # Invalid
        "/just/a/path",  # Invalid
        "http:///no_netloc", # Invalid (urlparse might treat this differently, let's test)
        "invalid-url" # Invalid
    ]

    for test_url in test_urls:
        try:
            scheme, netloc, path = parse_and_validate_url(test_url)
            print(f"Parsed URL: {test_url}")
            print(f"  Scheme: {scheme}")
            print(f"  Netloc: {netloc}")
            print(f"  Path: {path}\n")
        except ValueError as e:
            print(f"Error parsing URL '{test_url}': {e}\n")

    # Test case for urlparse behavior with "http:///no_netloc"
    # urlparse("http:///no_netloc") results in ParseResult(scheme='http', netloc='', path='/no_netloc', params='', query='', fragment='')
    # Our validation correctly catches this as netloc is empty.

    print("-" * 30)
    print("Testing build_url_with_query:")
    # Example Usage for build_url_with_query
    base_url = "http://www.example.com"
    search_path = "/search"
    query_params = {"q": "python programming", "page": 2, "sort_by": "relevance"}
    
    full_url = build_url_with_query(base_url, search_path, query_params)
    print(f"Base: {base_url}, Path: {search_path}, Params: {query_params}")
    print(f"Constructed URL: {full_url}\n")

    base_url_slash = "http://www.example.com/"
    path_slash = "/api/v1/users"
    params_empty: dict = {}
    full_url_no_params = build_url_with_query(base_url_slash, path_slash, params_empty)
    print(f"Base: {base_url_slash}, Path: {path_slash}, Params: {params_empty}")
    print(f"Constructed URL (no params): {full_url_no_params}\n")

    base_url_no_slash = "https://test.domain"
    path_no_slash = "items"
    params_special_chars = {"name": "A & B", "id": "123 456"}
    full_url_special_chars = build_url_with_query(base_url_no_slash, path_no_slash, params_special_chars)
    print(f"Base: {base_url_no_slash}, Path: {path_no_slash}, Params: {params_special_chars}")
    print(f"Constructed URL (special chars): {full_url_special_chars}\n")
