import re

_HTTP_URL_RE = re.compile(
    r"^(?:https?://)"
    r"(?:(?:localhost)|"
    r"(?:(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+(?:[a-z]{2,}|xn--[a-z0-9-]{2,}))|"
    r"(?:\d{1,3}(?:\.\d{1,3}){3}))"
    r"(?::\d{2,5})?"
    r"(?:/[^\s?#]*)?"
    r"(?:\?[^\s#]*)?"
    r"(?:#[^\s]*)?"
    r"$",
    re.IGNORECASE,
)

# RFC 3986 path-abempty (as used by http/https): "/" segments of pchar
_HTTP_PATH_RE = re.compile(
    r"^/(?:[A-Za-z0-9\-._~!$&'()*+,;=:@]|%[0-9A-Fa-f]{2})*(?:/(?:[A-Za-z0-9\-._~!$&'()*+,;=:@]|%[0-9A-Fa-f]{2})*)*$"
)

def is_http_url(url: str) -> bool:
    """
    Return True if the given url starts with 'http://' or 'https://'.
    Never raises; returns False for non-string inputs.
    """
    if not isinstance(url, str):
        return False
    return url.startswith(("http://", "https://"))

def validate_url(url: str) -> bool:
    """
    Return True if the given string is a syntactically valid HTTP/HTTPS URL,
    including domain names (or IPv4/localhost) and optional path, query, and fragment.
    Never raises; returns False for non-string inputs.
    """
    if not isinstance(url, str):
        return False
    return _HTTP_URL_RE.match(url) is not None

def validate_path(path: str) -> bool:
    """
    Return True if the given string is a syntactically valid HTTP/HTTPS path
    (path-abempty per RFC 3986), starting with '/' and containing valid pchars.
    Never raises; returns False for non-string inputs.
    """
    if not isinstance(path, str):
        return False
    return _HTTP_PATH_RE.fullmatch(path) is not None
