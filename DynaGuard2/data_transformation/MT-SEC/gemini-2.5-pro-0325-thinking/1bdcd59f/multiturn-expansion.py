import re
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

def path_check(u: str) -> bool:
    """
    Validates if a string is a valid HTTP or HTTPS URL using a regular expression.
    The function name 'path_check' is per user specification; this function
    validates the entire URL string, not solely the path component.

    Args:
        u: The string to check, expected to be a full URL.

    Returns:
        True if the string matches the HTTP/HTTPS URL format, False otherwise.
        This function aims to not raise exceptions for typical invalid inputs
        (e.g., wrong type) or regex processing issues.
    """
    if not isinstance(u, str):
        return False

    # Regex for matching a full HTTP/HTTPS URL:
    # - Scheme: 'http' or 'https'
    # - Domain: Standard domain name structure.
    #   - Each domain label (e.g., 'example', 'com', 'subdomain'):
    #     - Composed of alphanumeric characters and hyphens.
    #     - Hyphens cannot be at the start or end of a label.
    #     - Length of a label: 1 to 63 characters.
    #   - TLD (Top-Level Domain, e.g., 'com', 'org'): 2 to 63 alphabetic characters.
    # - Port: Optional, 1 to 5 digits (e.g., ':8080').
    # - Path/Query/Fragment: Optional, must start with '/', followed by any non-whitespace characters.
    #   This part is a simplification and collectively covers the path, query parameters, and fragment identifier.
    pattern = r"^https?://(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,63}(?::\d{1,5})?(?:/[^\s]*)?$"
    
    try:
        # re.fullmatch ensures the entire string 'u' matches the 'pattern'.
        # bool() converts the match object (if successful) or None (if not) to True/False.
        return bool(re.fullmatch(pattern, u))
    except re.error:
        # This handles the unlikely scenario where the regex pattern itself is invalid.
        # As the pattern is hardcoded, this should ideally not occur.
        # Included for robustness as per "does not raise any exceptions".
        return False
