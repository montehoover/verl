import re
from typing import List, Tuple

_URL_PATTERN = re.compile(
    r"""
    (?:
        (?P<scheme>https?|ftp)://[^\s<>'"()]+
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


def find_urls(text: str) -> List[Tuple[str, str]]:
    """
    Scan the input string and return a list of (url, scheme) tuples.
    Matches http(s)/ftp URLs and www.-style URLs. For www.-style URLs, the
    scheme is assumed to be 'http'.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    results: List[Tuple[str, str]] = []
    for m in _URL_PATTERN.finditer(text):
        raw = m.group(0)
        url = _rstrip_trailing_punct(raw)
        scheme = m.group('scheme')
        if scheme:
            scheme = scheme.lower()
        else:
            # Assume http for scheme-less www.* URLs
            if url.lower().startswith('www.'):
                scheme = 'http'
            else:
                scheme = ''
        results.append((url, scheme))
    return results
