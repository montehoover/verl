import re


def is_http_protocol(url: str) -> bool:
    """
    Return True if the given string starts with 'http://' or 'https://'.
    """
    if not isinstance(url, str):
        return False
    s = url.lstrip().lower()
    return s.startswith(('http://', 'https://'))


def extract_url_components(url: str) -> dict:
    """
    Extract components from a URL-like string.

    - protocol: The section of the URL before '://'
    - domain: The section of the URL between '://' and the next '/'
    - path: The trailing content after the domain (includes leading '/' if present)
    """
    if not isinstance(url, str):
        return {'protocol': '', 'domain': '', 'path': ''}

    s = url.strip()
    protocol = ''
    domain = ''
    path = ''

    sep = '://'
    sep_index = s.find(sep)
    if sep_index != -1:
        protocol = s[:sep_index]
        rest = s[sep_index + len(sep):]
    else:
        rest = s

    # Handle protocol-relative URLs beginning with '//example.com/...'
    if rest.startswith('//'):
        rest = rest[2:]

    slash_index = rest.find('/')
    if slash_index == -1:
        domain = rest
        path = ''
    else:
        domain = rest[:slash_index]
        path = rest[slash_index:]

    return {'protocol': protocol, 'domain': domain, 'path': path}


# Compiled regex for validating HTTP/HTTPS URLs
HTTP_URL_RE = re.compile(
    r'^'
    r'https?://'                                  # protocol
    r'(?:'                                        # host
    r'localhost'                                  # localhost
    r'|'
    r'(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?'   # label
    r'(?:\.(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?))*)'  # .label*
    r'|'
    r'(?:\d{1,3}\.){3}\d{1,3}'                    # IPv4 (loose)
    r'|'
    r'\[[0-9A-Fa-f:.]+\]'                         # IPv6 (loose, in brackets)
    r')'
    r'(?::\d{1,5})?'                              # optional port
    r'(?:/[^\s?#]*)?'                             # optional path
    r'(?:\?[^\s#]*)?'                             # optional query
    r'(?:#[^\s]*)?'                               # optional fragment
    r'$',
    re.IGNORECASE
)


def validate_path(path: str) -> bool:
    """
    Return True if the input is a well-formed http or https URL, otherwise False.
    Does not raise exceptions.
    """
    try:
        if not isinstance(path, str):
            return False
        s = path.strip()
        if not s:
            return False
        return HTTP_URL_RE.fullmatch(s) is not None
    except Exception:
        return False
