def construct_url(base_url: str, path: str) -> str:
    """
    Construct a URL by concatenating the base_url and path.

    Args:
        base_url (str): The base URL.
        path (str): The path to append to the base URL.

    Returns:
        str: The concatenated full URL.
    """
    return f"{base_url}{path}"
