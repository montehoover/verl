import re

HTTP_URL_REGEX = re.compile(
    r'^(https?://)'
    r'('
    r'(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}'      # domain like example.com
    r'|'
    r'(?:\d{1,3}\.){3}\d{1,3}'               # IPv4 address
    r'|'
    r'localhost'                             # localhost
    r')'
    r'(?:\:[0-9]{1,5})?'                     # optional port
    r'(?:/[^\s?#]*)?'                        # optional path
    r'(?:\?[^\s#]*)?'                        # optional query
    r'(?:\#[^\s]*)?'                         # optional fragment
    r'$',
    re.IGNORECASE
)

def check_path_validity(path_string: str) -> bool:
    """
    Validate whether the given string is a properly formatted HTTP or HTTPS URL.

    Args:
        path_string: The string to validate as an HTTP/HTTPS URL.

    Returns:
        True if the string is a valid HTTP/HTTPS URL, False otherwise.
    """
    if not isinstance(path_string, str):
        return False
    return HTTP_URL_REGEX.fullmatch(path_string) is not None
