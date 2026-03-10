def starts_with_http(url_string: str) -> bool:
    """
    Checks if a given string starts with 'http'.

    Args:
        url_string: The string to check.

    Returns:
        True if the string starts with 'http', False otherwise.
    """
    return url_string.startswith("http")


def is_http_or_https(url_string: str) -> bool:
    """
    Checks if a given string starts with 'http' or 'https'.

    Args:
        url_string: The string to check.

    Returns:
        True if the string starts with 'http' or 'https', False otherwise.
    """
    return url_string.startswith("http") or url_string.startswith("https")
