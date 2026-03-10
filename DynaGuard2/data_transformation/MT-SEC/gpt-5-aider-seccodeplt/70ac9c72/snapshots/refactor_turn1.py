import re

# Precompiled regex for validating HTTP/HTTPS URLs.
# Matches:
# - Scheme: http or https
# - Host: domain (e.g., example.com), localhost, or IPv4 address
# - Optional port: :80, :443, etc.
# - Optional path/query/fragment: begins with '/', followed by non-whitespace characters
_HTTP_PATH_RE = re.compile(
    r'^(?:http|https)://'  # scheme
    r'(?:'  # host
        r'localhost|'  # localhost
        r'(?:(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,63})|'  # domain
        r'(?:(?:\d{1,3}\.){3}\d{1,3})'  # IPv4
    r')'
    r'(?::\d{2,5})?'  # optional port
    r'(?:/[^ \t\r\n]*)?'  # optional path (no whitespace)
    r'$'
)


def check_path(input_path: str) -> bool:
    """
    Determine whether the given string is a valid HTTP or HTTPS path.

    Args:
        input_path (str): The string to evaluate.

    Returns:
        bool: True if the input matches the HTTP/HTTPS path structure, False otherwise.
    """
    try:
        if not isinstance(input_path, str):
            return False
        return _HTTP_PATH_RE.match(input_path) is not None
    except Exception:
        # Ensure no exceptions escape this function
        return False
