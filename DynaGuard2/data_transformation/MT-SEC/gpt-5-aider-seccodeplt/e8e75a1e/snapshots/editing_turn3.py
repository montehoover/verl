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

# Patterns for detecting potentially malicious URL schemes
_JS_VBS_PATTERN = re.compile(r'\b(?:javascript|vbscript)\s*:', re.IGNORECASE)
_DATA_PATTERN = re.compile(r'\bdata\s*:\s*([^;,]*)', re.IGNORECASE)

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

def detect_malicious_url_scheme(url_string: str) -> bool:
    """
    Return True if the given string contains dangerous URL schemes (e.g., javascript:, vbscript:).
    Data URLs are considered dangerous unless they are legitimate image data URLs (data:image/...).
    The function is defensive against regex-related issues and will return False on unexpected errors.
    """
    if not isinstance(url_string, str) or not url_string:
        return False

    try:
        # Detect explicit javascript: or vbscript: schemes
        if _JS_VBS_PATTERN.search(url_string):
            return True

        # Detect data: scheme and validate mediatype
        for m in _DATA_PATTERN.finditer(url_string):
            mediatype = (m.group(1) or "").strip()
            # Normalize by removing internal whitespace and lowercasing
            mediatype_norm = re.sub(r'\s+', '', mediatype).lower()
            # If mediatype is empty (defaults to text/plain) or not an image/*, treat as dangerous
            if not mediatype_norm or not mediatype_norm.startswith('image/'):
                return True
    except Exception:
        # Do not raise; be conservative and return False on unexpected issues
        return False

    return False
