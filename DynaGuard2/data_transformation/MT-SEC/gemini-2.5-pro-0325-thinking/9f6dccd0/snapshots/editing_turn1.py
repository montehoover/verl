def construct_url(base_url: str, path: str) -> str:
    """
    Constructs a URL by concatenating a base URL and a path.

    Args:
        base_url: The base URL (e.g., "http://example.com").
        path: The path component (e.g., "/users").

    Returns:
        The full URL.
    """
    return base_url + path
