from urllib.parse import urlparse

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
