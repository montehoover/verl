def starts_with_http(s: str) -> bool:
    """
    Return True if the given string starts with 'http', otherwise False.
    """
    return s.startswith("http")


def is_http_or_https(s: str) -> bool:
    """
    Return True if the given string starts with 'http' or 'https', otherwise False.
    """
    return s.startswith(("http", "https"))
