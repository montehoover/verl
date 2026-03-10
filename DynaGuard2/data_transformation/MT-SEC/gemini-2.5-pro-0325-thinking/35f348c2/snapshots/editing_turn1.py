def is_http_url(url: str) -> bool:
    """
    Checks if a given string starts with 'http://' or 'https://'.

    Args:
        url: The string to check.

    Returns:
        True if the url starts with 'http://' or 'https://', False otherwise.
    """
    return url.startswith('http://') or url.startswith('https://')
