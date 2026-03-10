from urllib.parse import urlparse, urlencode

def parse_and_validate_url(url: str):
    """
    Parses a URL and validates its basic structure.

    Args:
        url: The URL string to parse and validate.

    Returns:
        A tuple containing the scheme, netloc, and path of the URL.

    Raises:
        ValueError: If the URL is invalid (e.g., missing scheme or netloc).
    """
    parsed_url = urlparse(url)
    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError(f"Invalid URL: {url}. Scheme and netloc are required.")
    return parsed_url.scheme, parsed_url.netloc, parsed_url.path

def build_query_string(params: dict) -> str:
    """
    Builds a URL-encoded query string from a dictionary of parameters.

    Args:
        params: A dictionary of parameters (key-value pairs).

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
        "/just/a/path",
        "www.example.com/no/scheme"
    ]

    for u in test_urls:
        try:
            scheme, netloc, path = parse_and_validate_url(u)
            print(f"Parsed URL: {u}")
            print(f"  Scheme: {scheme}")
            print(f"  Netloc: {netloc}")
            print(f"  Path: {path}\n")
        except ValueError as e:
            print(f"Error parsing URL '{u}': {e}\n")

    # Example Usage for build_query_string
    params_dict = {
        "name": "John Doe",
        "age": 30,
        "city": "New York"
    }
    query_string = build_query_string(params_dict)
    print(f"Generated query string for {params_dict}: {query_string}\n")

    params_dict_with_special_chars = {
        "search": "python programming",
        "page": 2,
        "filter": "date&time"
    }
    query_string_special = build_query_string(params_dict_with_special_chars)
    print(f"Generated query string for {params_dict_with_special_chars}: {query_string_special}\n")
