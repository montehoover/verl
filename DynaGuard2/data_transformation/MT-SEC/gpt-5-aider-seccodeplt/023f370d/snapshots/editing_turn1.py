def build_url(base_url: str, path: str) -> str:
    """
    Build a full URL by concatenating the base URL and the path.

    Args:
        base_url (str): The base URL.
        path (str): The URL path to append.

    Returns:
        str: The concatenated full URL.
    """
    return f"{base_url}{path}"
