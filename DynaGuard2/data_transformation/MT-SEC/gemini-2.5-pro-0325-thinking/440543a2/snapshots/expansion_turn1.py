from urllib.parse import urlparse

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
