import re

# Compiled regex to validate HTTP/HTTPS URLs with optional port, path, query, and fragment.
_HTTP_URL_PATTERN = re.compile(
    r'^(?:http|https)://'                  # Scheme
    r'(?:'                                 # Host
        r'localhost'                       # - localhost
        r'|'                               # - or domain
        r'(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,63}'
        r'|'                               # - or IPv4
        r'(?:\d{1,3}\.){3}\d{1,3}'
    r')'
    r'(?::\d{2,5})?'                       # Optional port
    r'(?:/[^\s?#]*)*'                      # Optional path segments
    r'(?:\?[^\s#]*)?'                      # Optional query
    r'(?:#[^\s]*)?'                        # Optional fragment
    r'$',
    re.IGNORECASE
)

def check_path_validity(path_string: str) -> bool:
    """
    Verify if the given string is a valid HTTP or HTTPS path based on a regex pattern.

    Args:
        path_string (str): The input string to validate as a path.

    Returns:
        bool: True if the path is formatted correctly, False otherwise.
    """
    try:
        if not isinstance(path_string, str):
            return False
        return _HTTP_URL_PATTERN.fullmatch(path_string) is not None
    except Exception:
        # Ensure no exceptions escape from this function
        return False
