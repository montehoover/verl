def is_http_protocol(url: str) -> bool:
    """
    Checks if a URL starts with 'http' or 'https'.

    Args:
        url: The URL string to check.

    Returns:
        True if the URL starts with 'http' or 'https', False otherwise.
    """
    return url.startswith('http://') or url.startswith('https://')
