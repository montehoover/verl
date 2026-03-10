import re

_URL_REGEX = re.compile(
    r'^(?:http|https|ftp)://'                # scheme
    r'(?:\S+(?::\S*)?@)?'                    # optional user:pass@
    r'(?P<host>'                             # capture host as 'host'
        r'localhost'                         # localhost
        r'|'
        r'(?:(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63})'  # domain
        r'|'
        r'(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)'                      # IPv4
        r'(?:\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3})'
    r')'
    r'(?:\:\d{2,5})?'                        # optional port
    r'(?:[/?#][^\s]*)?'                      # optional path/query/fragment
    r'$', re.IGNORECASE
)

def is_valid_url(url: str) -> bool:
    try:
        if not isinstance(url, str):
            return False
        candidate = url.strip()
        if not candidate:
            return False
        return _URL_REGEX.match(candidate) is not None
    except Exception:
        return False

def extract_domain(url: str):
    try:
        if not isinstance(url, str):
            return False
        candidate = url.strip()
        if not candidate:
            return False
        match = _URL_REGEX.match(candidate)
        if not match:
            return False
        return match.group('host')
    except Exception:
        return False
