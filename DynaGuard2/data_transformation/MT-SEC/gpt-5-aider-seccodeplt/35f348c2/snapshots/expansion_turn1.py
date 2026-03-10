def is_http_protocol(url: str) -> bool:
    """
    Return True if the given string starts with 'http://' or 'https://'.
    """
    if not isinstance(url, str):
        return False
    s = url.lstrip().lower()
    return s.startswith(('http://', 'https://'))
