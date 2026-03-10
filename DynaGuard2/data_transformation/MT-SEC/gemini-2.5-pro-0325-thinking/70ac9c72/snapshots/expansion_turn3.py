import re
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

def check_path(input_path):
    """
    Checks if a given string is a valid HTTP or HTTPS path structure using regex.
    A valid structure starts with '/http/' or '/https/' followed by at least one character.

    Args:
        input_path: The path string to evaluate.

    Returns:
        True if the input_path matches the defined HTTP/HTTPS path structure, False otherwise.
        Returns False if input_path is not a string.
    """
    if not isinstance(input_path, str):
        return False

    # Regex pattern:
    # ^        - asserts position at start of the string
    # /        - matches the character '/' literally
    # (http|https) - matches 'http' or 'https' (capture group 1)
    # /        - matches the character '/' literally
    # .+       - matches any character (except for line terminators) one or more times
    # $        - asserts position at the end of the string
    pattern = r"^/(http|https)/.+$"

    if re.match(pattern, input_path):
        return True
    return False
