import re

def starts_with_http(url_string: str) -> bool:
    """
    Checks if a given string starts with 'http'.

    Args:
        url_string: The string to check.

    Returns:
        True if the string starts with 'http', False otherwise.
    """
    return url_string.startswith("http")


def is_http_or_https(url_string: str) -> bool:
    """
    Checks if a given string starts with 'http' or 'https'.

    Args:
        url_string: The string to check.

    Returns:
        True if the string starts with 'http' or 'https', False otherwise.
    """
    return url_string.startswith("http") or url_string.startswith("https")


def check_path_validity(path_string: str) -> bool:
    """
    Checks if a string is a well-formed HTTP or HTTPS URL path using regular expressions.

    Args:
        path_string: The string to validate.

    Returns:
        True if the string is a valid HTTP/HTTPS URL, False otherwise.
    """
    # Regex to validate HTTP/HTTPS URLs
    # Supports:
    # - http or https scheme
    # - Domain name (alphanumeric, dots, hyphens)
    # - Optional port number
    # - Optional path (alphanumeric, dots, hyphens, slashes)
    # - Optional query parameters (alphanumeric, dots, hyphens, ampersands, equals, percent encoding)
    # - Optional fragment identifier (alphanumeric, underscores)
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'([a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,6}'  # Domain name
        r'(:\d+)?'  # Optional port
        r'(/[\w\.-]*)*'  # Optional path
        r'(\?[\w\.-=&%]*)?'  # Optional query string
        r'(#\w*)?$'  # Optional fragment
    )
    if re.fullmatch(url_pattern, path_string):
        return True
    return False
