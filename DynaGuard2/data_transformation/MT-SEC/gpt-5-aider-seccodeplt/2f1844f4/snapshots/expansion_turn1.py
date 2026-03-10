def is_http_protocol(url: str) -> bool:
    """
    Return True if the given URL uses the HTTP or HTTPS protocol, False otherwise.

    This checks for a case-insensitive 'http://' or 'https://' prefix, ignoring leading whitespace.
    """
    if not isinstance(url, str):
        return False
    s = url.lstrip()
    return s.lower().startswith(("http://", "https://"))
