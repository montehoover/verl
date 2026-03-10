import re
from typing import List, Optional, Tuple

# Simple URL-like pattern: matches schemes (http, https, ftp) or "www."
_URL_PATTERN = re.compile(
    r"""
    (?:
        (?P<scheme>https?|ftp)://  # scheme
        |                          # or
        (?P<www>www\d{0,3}\.)      # www., www2., etc.
    )
    [^\s<>"'{}|\\^`]+              # run of URL characters
    """,
    re.IGNORECASE | re.VERBOSE,
)

def _strip_trailing_punctuation(url: str) -> str:
    # Remove common trailing punctuation that often attaches to URLs in text
    trailing = '.,;:!?\'")]}>'
    while url and url[-1] in trailing:
        ch = url[-1]
        # Keep balanced closers if they have matching openers within the URL
        if ch == ')' and url.count('(') >= url.count(')'):
            break
        if ch == ']' and url.count('[') >= url.count(']'):
            break
        if ch == '}' and url.count('{') >= url.count('}'):
            break
        url = url[:-1]
    return url

def find_urls(text: str) -> List[Tuple[str, Optional[str]]]:
    """
    Scan a string and return a list of (url, scheme) tuples.
    Recognizes URLs starting with http://, https://, ftp://, or www.
    The scheme is returned in lowercase when present; None when no explicit scheme (e.g., starts with www).
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    results: List[Tuple[str, Optional[str]]] = []
    for m in _URL_PATTERN.finditer(text):
        url = _strip_trailing_punctuation(m.group(0))
        if not url:
            continue
        scheme = m.group('scheme')
        scheme = scheme.lower() if scheme else None
        results.append((url, scheme))
    return results
