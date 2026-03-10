import re
from urllib.parse import urlsplit

def extract_path(url: str) -> str:
    """
    Extract the path component from a full URL, excluding query parameters and fragments.

    Examples:
    - https://example.com/foo/bar?x=1#y -> /foo/bar
    - https://example.com -> ""
    """
    parsed = urlsplit(url)
    return parsed.path

def validate_http_path(path: str) -> bool:
    """
    Return True if the given path starts with '/http' or '/https', otherwise False.

    Character handling details:
    - The check is case-sensitive (per RFC 3986, path segments are case-sensitive).
    - No decoding/normalization is performed; percent-encoded sequences are matched as-is.
    - Whitespace is not trimmed; leading/trailing spaces will cause the check to fail unless they
      are part of the path before '/http' or '/https'.

    Examples:
    - "/http/example.com/foo"   -> True
    - "/https/example.com/foo"  -> True
    - "/HTTP/example.com/foo"   -> False  (case-sensitive)
    - " /http/example.com/foo"  -> False  (leading space)
    - "/ftp/example.com/foo"    -> False
    """
    if not isinstance(path, str):
        return False
    return path.startswith("/http") or path.startswith("/https")

# Regex for validating HTTP/HTTPS paths embedded within a URL path.
# Format: "/http|https/<host>[:port][/path...]"
# - Host may be:
#   - hostname (labels of 1-63 alnum/hyphen, dot-separated), including "localhost"
#   - IPv4 address
#   - IPv6 address in brackets, e.g., [::1]
# - Port is optional (1-5 digits)
# - Path segments may contain RFC 3986 pchar (excluding '?' and '#') and percent-encoded bytes.
_HTTP_PATH_RE = re.compile(
    r'^/(?:http|https)/'
    r'(?:(?:\[[0-9A-Fa-f:.]+\])|'                   # IPv6 in brackets
    r'(?:\d{1,3}\.){3}\d{1,3}|'                     # IPv4
    r'(?:[A-Za-z0-9-]{1,63}\.)*[A-Za-z0-9-]{1,63})' # Hostname (incl. localhost)
    r'(?::\d{1,5})?'                                # Optional port
    r'(?:/(?:[A-Za-z0-9\-._~!$&\'()*+,;=:@]|%[0-9A-Fa-f]{2})*)*'  # Path segments
    r'$'
)

def check_path(input_path: str) -> bool:
    """
    Validate that input_path matches an HTTP/HTTPS path of the form:
    /http|https/<host>[:port][/path...]

    Rules:
    - Case-sensitive.
    - No query ('?') or fragment ('#') characters allowed.
    - Percent-encoded bytes must be of the form %HH (hex).
    - Returns False on non-strings or any unexpected condition.
    """
    if not isinstance(input_path, str):
        return False
    try:
        return _HTTP_PATH_RE.fullmatch(input_path) is not None
    except Exception:
        return False
