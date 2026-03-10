import re

# Precompiled regex for HTTP/HTTPS path validation
_PCT_ENC = r'%(?:[0-9A-Fa-f]{2})'
_PCHAR = r'(?:[A-Za-z0-9._~-]|[!$&\'()*+,;=]|:|@|' + _PCT_ENC + r')'
_SEGMENT = r'(?:' + _PCHAR + r'*)'
# Path with at least one leading slash, optional query and fragment
_PATH_REGEX = (
    r'^(?:/' + _SEGMENT + r')+'                 # path-abempty (at least "/")
    r'(?:\?(?:' + _PCHAR + r'|/|\?)*' + r')?'   # optional query
    r'(?:#(?:' + _PCHAR + r'|/|\?)*' + r')?$'   # optional fragment
)
_HTTP_PATH_RE = re.compile(_PATH_REGEX)

def path_check(u):
    """
    Return True if u is a valid HTTP/HTTPS path, False otherwise.
    Never raises exceptions.
    """
    try:
        if not isinstance(u, str):
            return False
        return _HTTP_PATH_RE.match(u) is not None
    except Exception:
        return False
