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

def validate_http_path(path_string):
    """
    Validates if a given path starts with '/http' or '/https'.

    Args:
        path_string: The path string to validate.

    Returns:
        True if the path starts with '/http' or '/https', False otherwise.
    """
    if not isinstance(path_string, str):
        return False  # Or raise an error, depending on desired behavior for non-string input
    return path_string.startswith('/http') or path_string.startswith('/https')
