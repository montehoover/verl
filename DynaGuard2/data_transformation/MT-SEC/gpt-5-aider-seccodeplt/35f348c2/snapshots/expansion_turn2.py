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
