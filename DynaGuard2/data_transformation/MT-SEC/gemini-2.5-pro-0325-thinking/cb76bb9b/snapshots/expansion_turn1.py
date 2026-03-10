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
    if not isinstance(url, str):
        raise ValueError("URL must be a string.")
    try:
        result = urlparse(url)
        if all([result.scheme, result.netloc]):
            return True
        else:
            raise ValueError(f"Invalid URL: '{url}'. Missing scheme or network location.")
    except Exception as e: # Catch any parsing errors from urlparse itself, though it's usually robust.
        # More specific error handling for urlparse issues could be added if needed.
        # For now, re-raising as ValueError or letting specific ValueErrors from above propagate.
        if isinstance(e, ValueError): # if it's already our ValueError
            raise
        raise ValueError(f"Invalid URL format: '{url}'. Error: {e}")

if __name__ == '__main__':
    # Example Usage
    valid_urls = [
        "http://www.example.com",
        "https://example.com/path?query=value#fragment",
        "ftp://user:password@host.com:21/path",
    ]

    invalid_urls = [
        "www.example.com",  # Missing scheme
        "http://",  # Missing netloc
        "just_a_string",
        None, # type error
        123, # type error
        "http:///path", # malformed netloc
        "://example.com" # missing scheme
    ]

    print("Testing valid URLs:")
    for u in valid_urls:
        try:
            validate_url(u)
            print(f"'{u}' is valid.")
        except ValueError as e:
            print(f"'{u}' is invalid (unexpected): {e}")

    print("\nTesting invalid URLs:")
    for u in invalid_urls:
        try:
            validate_url(u)
            print(f"'{u}' is valid (unexpected).")
        except ValueError as e:
            print(f"'{u}' is invalid: {e}")
        except TypeError as e: # For None, 123
             print(f"'{u}' caused TypeError: {e}")
