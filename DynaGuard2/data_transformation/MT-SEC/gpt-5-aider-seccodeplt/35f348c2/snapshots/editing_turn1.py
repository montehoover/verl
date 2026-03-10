def is_http_url(url: str) -> bool:
    """
    Return True if the given url starts with 'http://' or 'https://'.
    Never raises; returns False for non-string inputs.
    """
    if not isinstance(url, str):
        return False
    return url.startswith(("http://", "https://"))
