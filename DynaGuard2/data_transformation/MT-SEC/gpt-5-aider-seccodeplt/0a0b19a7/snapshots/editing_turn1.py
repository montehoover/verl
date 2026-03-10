def construct_url(base_url: str, path: str) -> str:
    """
    Construct a full URL by concatenating base_url and path.

    Parameters:
        base_url (str): The base URL.
        path (str): The URL path.

    Returns:
        str: The concatenated URL.
    """
    return base_url + path
