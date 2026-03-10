from urllib.parse import urlparse

def validate_and_parse_url(url: str) -> dict:
    """
    Validates and parses a URL string.

    Args:
        url: The URL string to validate and parse.

    Returns:
        A dictionary with 'scheme', 'domain', and 'path' if the URL is valid.

    Raises:
        ValueError: If the URL is invalid (e.g., missing scheme or domain).
    """
    if not isinstance(url, str):
        raise ValueError("Input URL must be a string.")

    parsed_url = urlparse(url)

    if not parsed_url.scheme:
        raise ValueError(f"Invalid URL '{url}': Missing scheme (e.g., http, https).")
    
    if not parsed_url.netloc:
        raise ValueError(f"Invalid URL '{url}': Missing domain name.")

    return {
        'scheme': parsed_url.scheme,
        'domain': parsed_url.netloc,
        'path': parsed_url.path
    }

if __name__ == '__main__':
    # Example Usage
    test_urls = [
        "https://www.example.com/path/to/resource",
        "http://localhost:8080/api/v1/users",
        "ftp://files.example.com/uploads/file.txt",
        "www.example.com/path",  # Invalid: missing scheme
        "https://",  # Invalid: missing domain
        "https://example.com", # Valid
        "http://example", # Valid (though 'example' might not resolve, it's structurally valid)
        12345 # Invalid type
    ]

    for test_url in test_urls:
        try:
            print(f"Processing URL: {test_url}")
            result = validate_and_parse_url(test_url)
            print(f"Parsed URL: {result}")
        except ValueError as e:
            print(f"Error: {e}")
        print("-" * 20)
