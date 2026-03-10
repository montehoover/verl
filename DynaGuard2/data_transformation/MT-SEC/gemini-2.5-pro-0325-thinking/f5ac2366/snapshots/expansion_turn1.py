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
    except Exception as e: # Catches errors from urlparse itself for malformed URLs
        raise ValueError(f"Invalid URL: {url}. Parsing failed: {e}")

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
        "http:///a", # malformed
        None, # Will be caught by type hinting if used, but good to test
        123,
    ]

    print("Testing valid URLs:")
    for u in valid_urls:
        try:
            if validate_url(u):
                print(f"'{u}' is valid.")
        except ValueError as e:
            print(f"Error validating '{u}': {e}")

    print("\nTesting invalid URLs:")
    for u in invalid_urls:
        try:
            # Ensure `u` is a string for the function, as per its type hint
            if not isinstance(u, str):
                print(f"Skipping non-string input: {u}")
                # Or raise a TypeError here if strict input type is desired before calling validate_url
                # raise TypeError(f"URL must be a string, got {type(u)}")
                # For now, we'll let validate_url handle it or pass it as is if it's a string
                if validate_url(str(u) if u is not None else ""): # Example of converting, though not ideal
                     print(f"'{u}' is valid (after potential conversion).") # Should not happen for these test cases
            elif validate_url(u):
                print(f"'{u}' is valid.") # Should not happen for these test cases
        except ValueError as e:
            print(f"Error validating '{u}': {e}")
        except TypeError as e: # If we added a TypeError for non-string inputs
            print(f"Input error for '{u}': {e}")
