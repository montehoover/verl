import re

_HTTP_PATH_REGEX = re.compile(
    r'^(?:https?)://'                # scheme: http or https
    r'(?:\S+(?::\S*)?@)?'            # optional user:pass@
    r'(?:'
        r'localhost'                 # localhost
        r'|'
        r'(?:(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63})'  # domain
        r'|'
        r'(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)'                      # IPv4
        r'(?:\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3})'
    r')'
    r'(?:\:\d{2,5})?'                # optional port
    r'(?:[/?#][^\s]*)?'              # optional path/query/fragment
    r'$', re.IGNORECASE
)

def verify_path_format(address: str) -> bool:
    try:
        if not isinstance(address, str):
            return False
        candidate = address.strip()
        if not candidate:
            return False
        return _HTTP_PATH_REGEX.match(candidate) is not None
    except Exception:
        return False
