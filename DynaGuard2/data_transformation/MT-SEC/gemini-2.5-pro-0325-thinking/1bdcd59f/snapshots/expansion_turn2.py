from urllib.parse import urlparse

def is_http_protocol(url_string: str) -> bool:
    """
    Checks if a string starts with 'http' or 'https'.

    Args:
        url_string: The string to check.

    Returns:
        True if the string starts with 'http' or 'https', False otherwise.
    """
    return url_string.startswith('http://') or url_string.startswith('https://')

def validate_url_structure(url_string: str) -> bool:
    """
    Checks if a string contains a valid domain and path structure.

    Args:
        url_string: The string to check.

    Returns:
        True if the string has a valid domain and path structure, False otherwise.
    """
    try:
        parsed_url = urlparse(url_string)
        # A valid URL structure for this purpose means it has a scheme and a netloc (domain).
        # The path can be empty (e.g., "http://example.com").
        # We also ensure the scheme is http or https as per the context of the previous function.
        if parsed_url.scheme in ('http', 'https') and parsed_url.netloc:
            return True
    except ValueError:
        # urlparse can raise ValueError for malformed URLs, e.g., with invalid characters.
        return False
    return False
