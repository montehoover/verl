import re
from typing import List

def extract_components(text: str) -> List[str]:
    """
    Extract components from text using regex:
    - Words (Unicode letters, allowing internal apostrophes, e.g., don't)
    - Numbers (integers and decimals, optional sign)
    - Special characters (single non-word, non-whitespace characters)

    Returns a list of components in the order they appear.
    """
    pattern = re.compile(
        r"(?:[+-]?(?:\d+(?:\.\d+)?|\.\d+))"   # numbers: integers or decimals with optional sign
        r"|(?:[^\W\d_]+(?:'[^\W\d_]+)*)"      # words: letters only (Unicode), allow internal apostrophes
        r"|(?:[^\w\s])",                      # special characters: punctuation/symbols (single char)
        re.UNICODE
    )
    return pattern.findall(text)

_SCHEME_RE = re.compile(r'^\s*([A-Za-z][A-Za-z0-9+.\-]*):(?=\/\/|$|[^\\\s])')

def identify_url_scheme(url: str) -> str:
    """
    Identify and return the URL scheme (e.g., 'http', 'https', 'ftp').
    Returns 'no_scheme' if no valid scheme is found.
    """
    m = _SCHEME_RE.match(url)
    return m.group(1).lower() if m else 'no_scheme'
