def starts_with_http(url_string: str) -> bool:
    """
    Checks if a given string starts with 'http'.

    Args:
        url_string: The string to check.

    Returns:
        True if the string starts with 'http', False otherwise.
    """
    return url_string.startswith("http")
