def generate_user_url(base_url: str, user_id: str) -> str:
    """
    Generates a URL with the user_id appended as a query parameter.

    Args:
        base_url: The base URL (e.g., "http://example.com/profile").
        user_id: The user's ID.

    Returns:
        A new URL string with the user_id as a query parameter
        (e.g., "http://example.com/profile?user_id=123").
    """
    return f"{base_url}?user_id={user_id}"
