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


# ---- Dangerous scheme detection ----

# Allow optional whitespace/control chars between letters to catch obfuscation like "java\nscript:"
_OBF_WS = r'[\s\x00-\x1f]*'


def _obfs(name: str) -> str:
    # Insert optional whitespace/control chars between each character
    return ''.join(ch + _OBF_WS for ch in name)


# Patterns to detect dangerous schemes
_JAVASCRIPT_SCHEME_RE = re.compile(r'(?i)\b' + _obfs('javascript') + r':')
_VBSCRIPT_SCHEME_RE = re.compile(r'(?i)\b' + _obfs('vbscript') + r':')
_DATA_SCHEME_RE = re.compile(r'(?i)\b' + _obfs('data') + r':')

# Safe image data URLs (ignore these)
_SAFE_DATA_IMAGE_RE = re.compile(
    r'(?i)\b'
    + _obfs('data')
    + r':'
    + _OBF_WS
    + r'image/(?:png|jpe?g|gif|webp|bmp|x-icon|ico)'
    r'(?:;[^,]*)?,'
)


def check_javascript_scheme(url_str: str) -> bool:
    """
    Return True if the input contains dangerous URL schemes such as
    'javascript:' or 'vbscript:', or 'data:' URLs that are not legitimate
    image data URLs. Returns False otherwise.
    """
    if not isinstance(url_str, str):
        return False

    # Directly detect javascript: or vbscript:
    if _JAVASCRIPT_SCHEME_RE.search(url_str) is not None:
        return True
    if _VBSCRIPT_SCHEME_RE.search(url_str) is not None:
        return True

    # Detect data: schemes and flag unless they are safe image data URLs
    for m in _DATA_SCHEME_RE.finditer(url_str):
        tail = url_str[m.start():]
        # If this particular data: occurrence is a safe image data URL, ignore it
        if _SAFE_DATA_IMAGE_RE.match(tail) is not None:
            continue
        # Otherwise, treat as dangerous
        return True

    return False
