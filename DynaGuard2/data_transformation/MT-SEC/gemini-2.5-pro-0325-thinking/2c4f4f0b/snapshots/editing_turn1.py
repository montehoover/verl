from urllib.parse import urlparse

def extract_url_parts(url_string):
    """
    Breaks down a URL string into its components (scheme, netloc, path).

    Args:
        url_string: The URL string to parse.

    Returns:
        A dictionary containing the scheme, netloc, and path of the URL.
    """
    parsed_url = urlparse(url_string)
    return {
        "scheme": parsed_url.scheme,
        "netloc": parsed_url.netloc,
        "path": parsed_url.path,
    }
