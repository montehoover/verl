import re

# Precompiled regex for validating HTTP/HTTPS URLs.
# Supports:
# - http:// or https://
# - optional userinfo (user or user:password) before '@'
# - host as domain with TLD (e.g., example.com), IPv4, or 'localhost'
# - optional :port (1-5 digits)
# - optional path, query, and fragment
_HTTP_URL_REGEX = re.compile(
    r'^(?:http|https)://'
    r'(?:[A-Za-z0-9._%+-]+(?::[^@]*)?@)?'
    r'(?:'
        r'localhost'
        r'|'
        r'(?:\d{1,3}(?:\.\d{1,3}){3})'
        r'|'
        r'(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,}'
    r')'
    r'(?::\d{1,5})?'
    r'(?:/[^\s?#]*)?'
    r'(?:\?[^\s#]*)?'
    r'(?:#[^\s]*)?'
    r'$',
    re.IGNORECASE
)

def check_path(input_path: str) -> bool:
    """
    Validate whether the given string is an HTTP or HTTPS URL.

    Args:
        input_path: The URL to be validated.

    Returns:
        True if input_path matches a valid HTTP/HTTPS URL format, False otherwise.
    """
    if not isinstance(input_path, str):
        return False
    return _HTTP_URL_REGEX.fullmatch(input_path) is not None
