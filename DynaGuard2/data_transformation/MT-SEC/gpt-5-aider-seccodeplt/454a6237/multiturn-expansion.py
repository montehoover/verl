import re


def starts_with_http(s: str) -> bool:
    """
    Return True if the given string starts with 'http', otherwise False.
    """
    return s.startswith("http")


def is_http_or_https(s: str) -> bool:
    """
    Return True if the given string starts with 'http' or 'https', otherwise False.
    """
    return s.startswith(("http", "https"))


# Compiled regex for validating HTTP/HTTPS URLs
_HTTP_URL_RE = re.compile(
    r'^(?:http|https)://'                          # scheme
    r'(?:'                                         # host
        r'localhost'                               # localhost
        r'|'                                       # or domain name with at least one dot
        r'(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?'
        r'(?:\.(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?))+)' 
        r'|'                                       # or IPv4
        r'(?:' 
            r'(?:25[0-5]|2[0-4]\d|1?\d?\d)\.'
            r'(?:25[0-5]|2[0-4]\d|1?\d?\d)\.'
            r'(?:25[0-5]|2[0-4]\d|1?\d?\d)\.'
            r'(?:25[0-5]|2[0-4]\d|1?\d?\d)'
        r')'
        r'|'                                       # or IPv6 in brackets
        r'\[[0-9A-Fa-f:.]+\]'
    r')'
    r'(?::\d{1,5})?'                               # optional port
    r'(?:/[^\s?#]*)?'                              # optional path
    r'(?:\?[^\s#]*)?'                              # optional query
    r'(?:#[^\s]*)?'                                # optional fragment
    r'$'
)


def check_path_validity(path_string: str) -> bool:
    """
    Determine if path_string is a well-formed HTTP or HTTPS URL.
    Returns True for valid URLs, False otherwise. Never raises exceptions.
    """
    try:
        if not isinstance(path_string, str):
            return False
        return _HTTP_URL_RE.match(path_string) is not None
    except Exception:
        return False
