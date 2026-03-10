import re

# Precompiled regex to validate http/https URLs.
# - Supports:
#   * Scheme: http or https
#   * Host: domain with at least one dot, localhost, or IPv4
#   * Optional port: :2-5 digits
#   * Optional path, query, and fragment
_HTTP_URL_PATTERN = re.compile(
    r'^(?:https?)://'                               # scheme
    r'(?:'                                          # host alternatives
    r'(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+'  # domain labels + dot
    r'[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?'         # final label (TLD)
    r'|localhost'                                   # or localhost
    r'|(?:\d{1,3}\.){3}\d{1,3}'                     # or IPv4 (basic)
    r')'
    r'(?::\d{2,5})?'                                # optional port
    r'(?:/(?:[^\s?#]*))?'                           # optional path
    r'(?:\?[^\s#]*)?'                               # optional query
    r'(?:\#[^\s]*)?'                                # optional fragment
    r'$', re.IGNORECASE
)

def check_path_validity(path_string: str) -> bool:
    """
    Verify if a string is a valid http or https URL/path using a regex.

    Args:
        path_string (str): The input string to validate as a path.

    Returns:
        bool: True for valid paths, False otherwise.
    """
    if not isinstance(path_string, str):
        return False

    candidate = path_string.strip()
    return _HTTP_URL_PATTERN.fullmatch(candidate) is not None
