from urllib.parse import urlparse

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
