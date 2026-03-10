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
