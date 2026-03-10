import re
from typing import List

# Simple URL-like pattern: matches schemes (http, https, ftp) or "www."
_URL_PATTERN = re.compile(
    r"""
    (?:
        (?:https?|ftp)://      # scheme
        |                      # or
        www\d{0,3}\.           # www., www2., etc.
    )
    [^\s<>"'{}|\\^`]+          # run of URL characters
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

def find_urls(text: str):  # -> List[str]
    """
    Scan a string and return a list of URL-like substrings.
    Recognizes URLs starting with http://, https://, ftp://, or www.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    matches = _URL_PATTERN.findall(text)
    results: List[str] = []
    for m in matches:
        cleaned = _strip_trailing_punctuation(m)
        if cleaned:
            results.append(cleaned)
    return results
