from urllib.parse import urlparse

def parse_url(url_string):
    """
    Parses a URL string and returns its scheme, netloc, and path.

    Args:
        url_string: The URL string to parse.

    Returns:
        A tuple containing the scheme, netloc, and path.
    """
    parsed_url = urlparse(url_string)
    return parsed_url.scheme, parsed_url.netloc, parsed_url.path
