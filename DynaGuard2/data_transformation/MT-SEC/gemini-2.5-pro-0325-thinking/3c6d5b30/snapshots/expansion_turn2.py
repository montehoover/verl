from urllib.parse import urlparse, urljoin

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
        # A valid URL must have a scheme (e.g., http, https) and a netloc (e.g., www.example.com)
        if all([result.scheme, result.netloc]):
            return True
        else:
            raise ValueError(f"Invalid URL: {url}. Scheme and network location are required.")
    except Exception as e: # Catch any parsing errors from urlparse itself or our custom ValueError
        if isinstance(e, ValueError): # Re-raise our custom ValueError
            raise
        # Wrap other exceptions in ValueError for consistent error handling
        raise ValueError(f"Invalid URL: {url}. Reason: {e}")

def concatenate_url_path(base_url: str, path: str) -> str:
    """
    Concatenates a base URL with a path.

    Args:
        base_url: The base URL (e.g., "http://www.example.com").
        path: The path to append (e.g., "/users/profile" or "users/profile").

    Returns:
        The full URL as a string.
    """
    if not validate_url(base_url): # Reuse existing validation
        # validate_url already raises ValueError, so we can just call it.
        # This line is more for explicit control flow, though validate_url would raise on its own.
        pass # Or raise a more specific error if needed, but validate_url's error is good.

    # urljoin handles slashes correctly (e.g., if base_url ends with / and path starts with /)
    return urljoin(base_url, path)

if __name__ == '__main__':
    # Example Usage
    valid_urls = [
        "http://www.example.com",
        "https://example.com/path?query=value#fragment",
        "ftp://user:password@host:port/path",
    ]

    invalid_urls = [
        "www.example.com",  # Missing scheme
        "http://",  # Missing netloc
        "just_a_string",
        None, # type: ignore
        123, # type: ignore
        "http:///path", # Missing netloc, but urlparse might still parse it
        "://example.com" # Missing scheme
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
        except TypeError as e: # Catching type errors for None/int inputs
             print(f"Correctly identified '{url_str}' as invalid (type error): {e}")

    # Test case for urlparse behavior with "http:///path"
    # urlparse("http:///path") -> ParseResult(scheme='http', netloc='', path='/path', params='', query='', fragment='')
    # Our validation should catch this.
    print("\nSpecific test for 'http:///path':")
    try:
        validate_url("http:///path")
        print("'http:///path' was unexpectedly considered valid.")
    except ValueError as e:
        print(f"Correctly identified 'http:///path' as invalid: {e}")

    print("\nTesting URL concatenation:")
    test_cases_concat = [
        ("http://www.example.com", "path/to/resource", "http://www.example.com/path/to/resource"),
        ("http://www.example.com/", "path/to/resource", "http://www.example.com/path/to/resource"),
        ("http://www.example.com", "/path/to/resource", "http://www.example.com/path/to/resource"),
        ("http://www.example.com/", "/path/to/resource", "http://www.example.com/path/to/resource"),
        ("http://www.example.com/api", "v1/users", "http://www.example.com/api/v1/users"),
        ("http://www.example.com/api/", "v1/users", "http://www.example.com/api/v1/users"),
        ("http://www.example.com/api", "/v1/users", "http://www.example.com/api/v1/users"),
        ("http://www.example.com/api/", "/v1/users", "http://www.example.com/api/v1/users"),
        ("https://example.com/test/", "../another", "https://example.com/another"), # Relative path
        ("http://www.example.com", "path with spaces", "http://www.example.com/path%20with%20spaces"), # Path encoding
    ]

    for base, path_segment, expected in test_cases_concat:
        try:
            result = concatenate_url_path(base, path_segment)
            if result == expected:
                print(f"concatenate_url_path('{base}', '{path_segment}') == '{result}' (Correct)")
            else:
                print(f"concatenate_url_path('{base}', '{path_segment}') == '{result}' (Incorrect, expected '{expected}')")
        except ValueError as e:
            print(f"Error concatenating '{base}' and '{path_segment}': {e}")

    print("\nTesting URL concatenation with invalid base URL:")
    try:
        concatenate_url_path("not_a_valid_base", "path")
        print("Concatenation with invalid base URL did not raise ValueError as expected.")
    except ValueError as e:
        print(f"Correctly caught error for invalid base URL: {e}")
