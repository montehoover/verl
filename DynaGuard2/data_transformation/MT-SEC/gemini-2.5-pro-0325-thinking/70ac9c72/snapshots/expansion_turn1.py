from urllib.parse import urlparse

def extract_path(url_string):
    """
    Extracts the path component from a URL string.

    Args:
        url_string: The full URL as a string.

    Returns:
        The path component of the URL.
    """
    parsed_url = urlparse(url_string)
    return parsed_url.path
