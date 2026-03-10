from urllib.parse import urlparse

def validate_url(url: str) -> bool:
    """
    Validates a given URL.

    Args:
        url: The URL string to validate.

    Returns:
        True if the URL is valid.

    Raises:
        ValueError: If the URL is invalid.
    """
    try:
        result = urlparse(url)
        if all([result.scheme, result.netloc]):
            return True
        else:
            raise ValueError(f"Invalid URL: {url}. Missing scheme or network location.")
    except Exception as e: # Catch any parsing errors from urlparse itself, though less common for basic structure
        raise ValueError(f"Invalid URL: {url}. Parsing error: {e}")

if __name__ == '__main__':
    # Example Usage
    valid_urls = [
        "http://www.example.com",
        "https://example.com/path?query=value#fragment",
        "ftp://user:password@host:port/path",
    ]
    invalid_urls = [
        "www.example.com",
        "example.com",
        "http//example.com",
        "just_a_string",
        "",
        None, # type: ignore
        "http://",
        "http:///path",
    ]

    print("Testing valid URLs:")
    for url_str in valid_urls:
        try:
            if validate_url(url_str):
                print(f"'{url_str}' is valid.")
        except ValueError as e:
            print(f"Error validating '{url_str}': {e}")

    print("\nTesting invalid URLs:")
    for url_str in invalid_urls:
        try:
            validate_url(url_str)
            print(f"'{url_str}' was unexpectedly considered valid.")
        except ValueError as e:
            print(f"Correctly identified '{url_str}' as invalid: {e}")
        except TypeError as e: # Handles None case specifically if type hints are enforced at runtime
             print(f"Correctly identified '{url_str}' as invalid (TypeError): {e}")
