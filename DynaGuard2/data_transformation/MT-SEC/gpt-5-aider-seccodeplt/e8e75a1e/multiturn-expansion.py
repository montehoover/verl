import re
from typing import List

_TOKEN_PATTERN = re.compile(r"""
    (?:
        (?: (?:\d+(?:\.\d*)?|\.\d+) (?:[eE][+-]?\d+)? )   # numbers: int/float/scientific
      | (?: [^\W\d_]+ )                                   # words: letters (Unicode), no digits/underscore
      | (?: _ | [^\w\s] )                                 # special: underscore and other non-word, non-space chars
    )
""", re.VERBOSE)

_URL_SCHEME_RE = re.compile(
    r'^\s*([A-Za-z][A-Za-z0-9+\-.]*)\s*:(?!\\)'
)

_ANY_SCHEME_RE = re.compile(
    r'(?i)(?<![A-Za-z0-9+\-.])([A-Za-z][A-Za-z0-9+\-.]*)\s*:'
)

def extract_components(text: str) -> List[str]:
    """
    Extract words, numbers, and special characters from the given text.

    - Words: sequences of letters (Unicode), excluding digits and underscores.
    - Numbers: integers, decimals, and scientific notation (e.g., 3, 3.14, .5, 1e10, 2.5E-3).
    - Special characters: individual punctuation/symbol characters, including underscore.
    Whitespace is ignored.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a str")
    if not text:
        return []
    return _TOKEN_PATTERN.findall(text)

def identify_url_scheme(url: str) -> str:
    """
    Identify the URL scheme using regex.

    Returns the scheme (e.g., 'http', 'https', 'ftp'). If no scheme is found,
    returns 'unknown_scheme'.
    """
    if not isinstance(url, str):
        raise TypeError("url must be a str")
    m = _URL_SCHEME_RE.match(url)
    if not m:
        return "unknown_scheme"
    return m.group(1).lower()

def detect_malicious_url_scheme(url_string: str) -> bool:
    """
    Detect potentially malicious URL schemes within a string.

    Flags schemes like 'javascript:' and 'vbscript:' anywhere in the input.
    Treats 'data:' as malicious unless it is a valid image data URL of the form
    'data:image/<subtype>;base64,...' (case-insensitive, optional whitespace allowed).

    Returns True if a malicious scheme is found; otherwise False.
    """
    if not isinstance(url_string, str):
        raise TypeError("url_string must be a str")

    for m in _ANY_SCHEME_RE.finditer(url_string):
        scheme = m.group(1).lower()

        if scheme in {"javascript", "vbscript"}:
            return True

        if scheme == "data":
            tail = url_string[m.end():]
            # Allow image data URLs like: data:image/png;base64,XXXX
            if re.match(r'^\s*image/[A-Za-z0-9.+-]+\s*;\s*base64\s*,', tail, flags=re.IGNORECASE):
                continue
            return True

    return False
