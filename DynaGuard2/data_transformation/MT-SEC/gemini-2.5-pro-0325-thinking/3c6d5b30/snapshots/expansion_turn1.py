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
