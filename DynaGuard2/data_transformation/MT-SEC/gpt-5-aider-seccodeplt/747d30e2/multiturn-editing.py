import re
from typing import List, Optional, Tuple

_URL_REGEX = re.compile(
    r"""
    (?P<url>
        (?:
            (?P<scheme>[a-z][a-z0-9+.\-]*)://[^\s<>'"()]+     # scheme URLs like http://, https://, ftp://
            |
            www\.[^\s<>'"()]+                                 # www. prefixed URLs
            |
            \b(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/[^\s<>'"()]*)?    # bare domains with optional path
        )
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


def find_urls(text: str):  # -> List[Tuple[str, Optional[str]]]
    """
    Scan the input text and return a list of tuples (url, scheme).

    - scheme-based URLs (e.g., http://, https://, ftp://) will include their scheme (lowercased).
    - www-prefixed and bare domain URLs will have scheme set to None.

    This function does not validate URLs; it only extracts URL-like patterns.
    """
    if not isinstance(text, str):
        raise TypeError("find_urls expected 'text' to be a string")
    results: List[Tuple[str, Optional[str]]] = []
    for match in _URL_REGEX.finditer(text):
        candidate = match.group("url")
        cleaned = _strip_surrounding_punct(candidate)
        if cleaned:
            scheme = match.group("scheme")
            scheme = scheme.lower() if scheme else None
            results.append((cleaned, scheme))
    return results


# Precompiled regex to detect potentially malicious URL schemes
_MALICIOUS_SCHEME_RE = re.compile(
    r'(?<![a-z0-9+.\-])(?:javascript|jscript|vbscript)\s*:',
    re.IGNORECASE,
)


def is_javascript_scheme(s: str) -> bool:
    """
    Return True if the string contains a potentially malicious URL scheme
    such as 'javascript', 'jscript', or 'vbscript'. Image data URLs
    (e.g., data:image/png;base64,...) are not considered malicious by this check.
    """
    try:
        if not isinstance(s, str):
            return False
        return _MALICIOUS_SCHEME_RE.search(s) is not None
    except re.error:
        return False
