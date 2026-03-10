import re

# Precompiled regex for URL validation with host capture
_URL_RE = re.compile(
    r'^'
    r'(?:(?:https?|ftp)://)'  # scheme
    r'(?:[^\s/?#@]+(?::[^\s/?#@]*)?@)?'  # optional user:pass@
    r'(?P<host>'  # capture host
        r'localhost'  # localhost
        r'|'
        r'(?:'  # IPv4
            r'(?:25[0-5]|2[0-4]\d|1?\d?\d)'
            r'(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}'
        r')'
        r'|'
        r'(?:'  # domain
            r'(?!-)[A-Za-z0-9-]{1,63}(?<!-)'
            r'(?:\.(?!-)[A-Za-z0-9-]{1,63}(?<!-))*'
            r'\.[A-Za-z]{2,63}'
        r')'
    r')'
    r'(?::\d{2,5})?'  # optional port
    r'(?:[/?#][^\s]*)?'  # optional path/query/fragment
    r'$',
    re.IGNORECASE
)

def validate_url(url) -> bool:
    """
    Validate whether the given string is a well-formed URL using regex.
    Returns True if valid, False otherwise. Never raises exceptions.
    """
    try:
        if not isinstance(url, str):
            return False
        s = url.strip()
        if not s:
            return False
        return _URL_RE.match(s) is not None
    except Exception:
        return False

def extract_domain(url):
    """
    Return the domain/host part of the URL if it's valid, otherwise False.
    Never raises exceptions.
    """
    try:
        if not isinstance(url, str):
            return False
        s = url.strip()
        if not s:
            return False
        m = _URL_RE.match(s)
        if not m:
            return False
        return m.group('host')
    except Exception:
        return False
