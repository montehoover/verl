import re
from typing import List, Optional, Tuple

URL_PATTERN = re.compile(r"""
\b
(
    (?:(?P<scheme>https?|ftp)://)[^\s<>"']+       # scheme-based URLs
  |
    (?:www\.)[^\s<>"']+                           # www. URLs
  |
    (?<!@)(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,63}(?:/[^\s<>"']*)?  # bare domains with optional path, not emails
)
""", re.IGNORECASE | re.VERBOSE)

# Patterns to detect dangerous schemes and handle data URLs
DANGEROUS_SCHEME_RE = re.compile(r'\b(?:javascript|vbscript)\s*:', re.IGNORECASE)
DATA_SCHEME_RE = re.compile(r'\bdata\s*:', re.IGNORECASE)
IMAGE_DATA_PREFIX_RE = re.compile(r'\bdata\s*:\s*image\/[a-z0-9.+-]+', re.IGNORECASE)

def _trim_url(u: str) -> str:
    # Trim trailing punctuation and unmatched closing brackets often adjacent in prose.
    if not u:
        return u
    # First remove surrounding angle brackets if present <...>
    if u[0] == '<' and u.endswith('>'):
        u = u[1:-1]
    # Iteratively strip common trailing punctuation or unmatched closers
    closing_pairs = {')': '(', ']': '[', '}': '{'}
    while u:
        original = u
        # strip common trailing punctuation
        u = u.rstrip('.,!?:;')
        # strip trailing quotes
        u = u.rstrip('\'"')
        # strip unmatched closing brackets
        if u and u[-1] in closing_pairs:
            close = u[-1]
            open_ch = closing_pairs[close]
            if u.count(open_ch) < u.count(close):
                u = u[:-1]
        if u == original:
            break
    return u

def find_urls(text: str) -> List[Tuple[str, Optional[str]]]:
    """
    Scan the input text and return a list of (url, scheme) tuples.
    Matches scheme URLs (http, https, ftp), www-prefixed URLs, and bare domains with optional paths.
    Trailing punctuation commonly found in prose is trimmed from the results.
    For URLs without an explicit scheme (e.g., www.example.com or example.com/path), scheme is None.
    """
    if not isinstance(text, str) or not text:
        return []
    results: List[Tuple[str, Optional[str]]] = []
    for m in URL_PATTERN.finditer(text):
        candidate = m.group(0)
        cleaned = _trim_url(candidate)
        if cleaned:
            scheme = m.group('scheme')
            scheme = scheme.lower() if scheme else None
            results.append((cleaned, scheme))
    return results

def detect_js_scheme(url_string: str) -> bool:
    """
    Detect potentially harmful URL schemes within a string.
    Returns True if the string contains:
      - javascript: or vbscript: schemes (case-insensitive), or
      - data: URLs that are not image data (i.e., not data:image/...)
    Valid image data URLs are ignored (do not trigger detection).
    """
    if not isinstance(url_string, str) or not url_string:
        return False

    try:
        if DANGEROUS_SCHEME_RE.search(url_string):
            return True
    except re.error:
        return False

    try:
        for m in DATA_SCHEME_RE.finditer(url_string):
            tail = url_string[m.start():]
            # If this data: occurrence is not an image data URL, consider it dangerous
            if not IMAGE_DATA_PREFIX_RE.match(tail):
                return True
    except re.error:
        return False

    return False
