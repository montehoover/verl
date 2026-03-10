import re

_HTTP_URL_RE = re.compile(
    r'^https?://'
    r'(?:'
    r'(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}'
    r'|'
    r'localhost'
    r'|'
    r'(?:\d{1,3}\.){3}\d{1,3}'
    r')'
    r'(?::\d{1,5})?'
    r'(?:/[^\s?#]*)?'
    r'(?:\?[^\s#]*)?'
    r'(?:#[^\s]*)?'
    r'$',
    re.IGNORECASE
)

def verify_path_format(address: str) -> bool:
    try:
        if not isinstance(address, str):
            return False
        return _HTTP_URL_RE.match(address) is not None
    except Exception:
        return False
