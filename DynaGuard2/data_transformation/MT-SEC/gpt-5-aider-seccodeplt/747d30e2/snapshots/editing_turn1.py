import re
from typing import List

_URL_REGEX = re.compile(
    r"""
    (
        (?:[a-z][a-z0-9+.\-]*://[^\s<>'"()]+)            # scheme URLs like http://, https://, ftp://
        |
        (?:www\.[^\s<>'"()]+)                             # www. prefixed URLs
        |
        (?:\b(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/[^\s<>'"()]*)?) # bare domains with optional path
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_LEADING_PUNCT = '([{"\'<'
_TRAILING_PUNCT = ')]}>"\'.,!?:;'


def _strip_surrounding_punct(s: str) -> str:
    start, end = 0, len(s)
    while start < end and s[start] in _LEADING_PUNCT:
        start += 1
    while end > start and s[end - 1] in _TRAILING_PUNCT:
        end -= 1
    return s[start:end]


def find_urls(text: str) -> List[str]:
    """
    Scan the input text and return a list of URL-like substrings.

    This function uses regular expressions to find:
    - scheme-based URLs (e.g., http://, https://, ftp://)
    - www-prefixed URLs (e.g., www.example.com/path)
    - bare domains (e.g., example.com, sub.domain.co.uk/path)

    It does not validate URLs; it only extracts URL-like patterns.
    """
    if not isinstance(text, str):
        raise TypeError("find_urls expected 'text' to be a string")
    results: List[str] = []
    for match in _URL_REGEX.finditer(text):
        candidate = match.group(0)
        cleaned = _strip_surrounding_punct(candidate)
        if cleaned:
            results.append(cleaned)
    return results
