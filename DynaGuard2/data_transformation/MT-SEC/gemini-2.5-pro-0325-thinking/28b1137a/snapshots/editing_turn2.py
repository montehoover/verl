def build_user_url(base_url: str, path: str, user_id: str) -> str:
    """
    Constructs a URL by concatenating a base URL, a path, and a user_id as a query parameter.

    Args:
        base_url: The base URL (e.g., "http://example.com").
        path: The path to append (e.g., "/users").
        user_id: The user ID to include as a query parameter.

    Returns:
        The full URL with the user_id query parameter.
    """
    return f"{base_url}{path}?user_id={user_id}"
