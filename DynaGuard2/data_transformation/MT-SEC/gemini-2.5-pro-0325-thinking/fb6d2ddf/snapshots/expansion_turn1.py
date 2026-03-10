from urllib.parse import urlparse

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
