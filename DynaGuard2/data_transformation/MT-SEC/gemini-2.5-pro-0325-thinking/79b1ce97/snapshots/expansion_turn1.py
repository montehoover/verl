from urllib.parse import urlparse

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
        raise ValueError(f"Invalid URL: '{url}'. URL must have a scheme and netloc.")
    
    return {
        "scheme": parsed_url.scheme,
        "netloc": parsed_url.netloc,
        "path": parsed_url.path,
    }

if __name__ == '__main__':
    # Example Usage
    test_urls = [
        "http://www.example.com/path/to/resource",
        "https://subdomain.example.co.uk:8080/another/path?query=param#fragment",
        "ftp://user:pass@example.com/dir/",
        "invalid-url",
        "www.missing-scheme.com",
        "http:///missing-netloc",
    ]

    for u in test_urls:
        try:
            components = parse_and_validate_url(u)
            print(f"Parsed '{u}': {components}")
        except ValueError as e:
            print(f"Error parsing '{u}': {e}")

    # Example of accessing components
    try:
        valid_url = "https://docs.python.org/3/library/urllib.parse.html"
        parsed = parse_and_validate_url(valid_url)
        print(f"\nScheme of '{valid_url}': {parsed['scheme']}")
        print(f"Netloc of '{valid_url}': {parsed['netloc']}")
        print(f"Path of '{valid_url}': {parsed['path']}")
    except ValueError as e:
        print(f"Error: {e}")
