import re
from typing import List

_URL_PATTERN = re.compile(
    r"""
    (?:
        (?:https?|ftp)://[^\s<>'"()]+
        |
        (?<!@)\bwww\.[^\s<>'"()]+
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_TRAILING_PUNCT = ".,;:!?)]}'\""


def _rstrip_trailing_punct(url: str) -> str:
    # Strip common trailing punctuation that is unlikely to be part of the URL
    while url and url[-1] in _TRAILING_PUNCT:
        ch = url[-1]
        if ch == ')' and url.count(')') <= url.count('('):
            break
        if ch == ']' and url.count(']') <= url.count('['):
            break
        if ch == '}' and url.count('}') <= url.count('{'):
            break
        url = url[:-1]
    return url


def find_urls(text: str) -> List[str]:
    """
    Scan the input string and return a list of URL-like substrings.
    Matches http(s)/ftp URLs and www.-style URLs.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    urls: List[str] = []
    for m in _URL_PATTERN.finditer(text):
        url = m.group(0)
        url = _rstrip_trailing_punct(url)
        urls.append(url)
    return urls
