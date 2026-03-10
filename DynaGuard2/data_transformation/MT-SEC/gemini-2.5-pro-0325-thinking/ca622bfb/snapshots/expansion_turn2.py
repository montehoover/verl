from urllib.parse import urlparse

def validate_url(url: str) -> bool:
    """
    Validates a URL string.

    Args:
        url: The URL string to validate.

    Returns:
        True if the URL is valid.

    Raises:
        ValueError: If the URL is not valid (e.g., missing scheme or netloc).
    """
    try:
        parsed_url = urlparse(url)
        if not parsed_url.scheme:
            raise ValueError(f"URL '{url}' is missing a scheme (e.g., http, https).")
        if not parsed_url.netloc:
            raise ValueError(f"URL '{url}' is missing a network location (e.g., www.example.com).")
        return True
    except Exception as e: # Catching potential errors from urlparse itself, though less common for basic validation
        raise ValueError(f"URL '{url}' is malformed: {e}")

def build_url_with_path(base_url: str, path_component: str) -> str:
    """
    Combines a base URL with a path component, ensuring correct slash handling.

    Args:
        base_url: The base URL string.
        path_component: The path component to append.

    Returns:
        The combined URL string.
    """
    # Ensure base_url ends with a slash if it doesn't have one and path_component doesn't start with one.
    # urljoin handles this logic well.
    # If path_component is an absolute path (starts with '/'), urljoin will treat it as such.
    return urlparse(base_url)._replace(path=urlparse(base_url).path.rstrip('/') + '/' + path_component.lstrip('/')).geturl()


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
        "://example.com", # Malformed scheme
        "justastring",
        None, # type error, but good to see how it's handled
        123, # type error
    ]

    print("Validating URLs:")
    for url_str in valid_urls:
        try:
            validate_url(url_str)
            print(f"'{url_str}' is valid.")
        except ValueError as e:
            print(f"Error validating '{url_str}': {e}")

    print("\nValidating Invalid URLs:")
    for url_str in invalid_urls:
        try:
            validate_url(str(url_str) if url_str is not None else "") # Ensure string for testing
            print(f"'{url_str}' was unexpectedly considered valid.")
        except ValueError as e:
            print(f"Correctly invalidated '{url_str}': {e}")
        except TypeError as e: # Catching type errors for non-string inputs
             print(f"Correctly caught TypeError for input '{url_str}': {e}")

    print("\nBuilding URLs with path components:")
    test_builds = [
        ("http://example.com", "path1"),
        ("http://example.com/", "path2"),
        ("http://example.com", "/path3"),
        ("http://example.com/", "/path4"),
        ("http://example.com/api", "endpoint"),
        ("http://example.com/api/", "endpoint"),
        ("http://example.com/api", "/endpoint"),
        ("http://example.com/api/", "/endpoint/"),
        ("http://example.com/api/", "v1/resource"),
        ("http://example.com/api", "v1/resource/"),
    ]
    for base, path in test_builds:
        try:
            # First validate the base_url to ensure it's a good starting point
            validate_url(base)
            combined_url = build_url_with_path(base, path)
            # Validate the combined URL as well
            validate_url(combined_url)
            print(f"Base: '{base}', Path: '{path}' -> Combined: '{combined_url}'")
        except ValueError as e:
            print(f"Error building or validating URL (Base: '{base}', Path: '{path}'): {e}")
