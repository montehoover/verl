from urllib.parse import urlparse

def parse_image_url(url_string):
    """
    Parses a URL string and returns its components.

    Args:
        url_string: The URL string to parse.

    Returns:
        A tuple containing the scheme, netloc, path, and query.
    """
    parsed_url = urlparse(url_string)
    return parsed_url.scheme, parsed_url.netloc, parsed_url.path, parsed_url.query
