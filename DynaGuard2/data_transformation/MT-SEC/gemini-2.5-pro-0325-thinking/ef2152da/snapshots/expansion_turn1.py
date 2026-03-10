from urllib.parse import urlparse

def extract_url_parts(url_string):
    """
    Parses a URL string and returns its scheme, netloc, and path.

    Args:
        url_string (str): The URL to parse.

    Returns:
        tuple: A tuple containing the scheme, netloc, and path of the URL.
               Returns (None, None, None) if parsing fails, input is not a string,
               or if the URL is malformed in a way that urlparse cannot handle gracefully
               for these specific attributes.
    """
    if not isinstance(url_string, str):
        return None, None, None
    try:
        parsed_url = urlparse(url_string)
        # Ensure that even if parsing happens, if scheme or netloc are empty
        # (e.g. for relative paths or malformed URLs like "www.example.com"),
        # we treat it consistently. urlparse might return empty strings.
        # For this function's contract, we'll return them as is.
        scheme = parsed_url.scheme
        netloc = parsed_url.netloc
        path = parsed_url.path
        return scheme, netloc, path
    except Exception:
        # This handles unexpected errors during parsing, though urlparse is quite robust.
        return None, None, None
