def is_http_protocol(url_string: str) -> bool:
    """
    Checks if a given string starts with 'http' or 'https'.

    Args:
        url_string: The string to check.

    Returns:
        True if the string starts with 'http' or 'https', False otherwise.
    """
    return url_string.startswith('http://') or url_string.startswith('https://')
