import re
from typing import List

__all__ = ["extract_url_candidates", "identify_url_scheme", "detect_js_scheme"]

# Regex to find URL-like patterns:
# - Protocol-based: http(s):// or ftp://
# - "www." prefixed hostnames
# - Bare domains with TLDs (e.g., example.com, sub.example.co.uk), optional path
URL_PATTERN = re.compile(
    r"""
    \b(
        (?:                                     # protocol-based
            (?:(?:https?|ftp)://)
            [^\s<>\[\](){}"']+
        )
        |
        (?:                                     # www.-prefixed
            www\.
            [^\s<>\[\](){}"']+
        )
        |
        (?:                                     # bare domains with TLD
            (?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+(?:[a-z]{2,})
            (?:/[^\s<>\[\](){}"']*)?
        )
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Characters commonly trailing after URLs in prose that we want to trim.
_TRIM_TAIL_CHARS = '.,;:!?\'"])}>'
_BRACKET_PAIRS = {")": "(", "]": "[", "}": "{", ">": "<"}

# Regex to capture the scheme at the beginning of a URL candidate (e.g., "http", "https", "ftp", "mailto")
_SCHEME_REGEX = re.compile(r'^\s*([a-z][a-z0-9+.\-]*)\s*:', re.IGNORECASE)

# Regex to detect harmful script-like schemes within arbitrary strings.
_HARMFUL_SCHEME_RE = re.compile(r'(?i)(?:^|[^a-z0-9+.\-])(?:javascript|vbscript)\s*:\s*')

# Regex to find occurrences of the data: scheme.
_DATA_SCHEME_RE = re.compile(r'(?i)(?:^|[^a-z0-9+.\-])data\s*:\s*')

# Regex to match safe image data URLs starting at the given position (requires a comma to delimit the header).
_SAFE_IMAGE_DATA_START_RE = re.compile(
    r'^\s*data\s*:\s*image/[a-z0-9.+\-]+(?:\s*;[a-z0-9.+\-]+=[^;,]+)?(?:\s*;base64)?\s*,',
    re.IGNORECASE,
)


def _trim_trailing_punctuation(candidate: str) -> str:
    """
    Trim trailing punctuation and unbalanced closing brackets from a matched URL candidate.
    """
    s = candidate
    while s:
        changed = False

        # Remove unmatched closing brackets at the end (e.g., "...)" or "...]]")
        for closer, opener in _BRACKET_PAIRS.items():
            while s.endswith(closer) and s.count(opener) < s.count(closer):
                s = s[:-1]
                changed = True

        # Remove generic trailing punctuation characters
        while s and s[-1] in _TRIM_TAIL_CHARS:
            s = s[:-1]
            changed = True

        if not changed:
            break

    return s


def extract_url_candidates(text: str) -> List[str]:
    """
    Extract URL-like patterns from the given text.

    Args:
        text: Input text.

    Returns:
        A list of URL-like strings detected within the text.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a str")

    candidates: List[str] = []
    for m in URL_PATTERN.finditer(text):
        candidate = _trim_trailing_punctuation(m.group(0))
        if candidate:
            candidates.append(candidate)
    return candidates


def identify_url_scheme(url_candidate: str) -> str:
    """
    Identify the scheme of a URL candidate using regex.

    Args:
        url_candidate: A string that may represent a URL.

    Returns:
        The scheme (e.g., 'http', 'https', 'ftp', 'mailto', etc.) in lowercase.
        If no scheme is detected, returns 'unknown'.
    """
    if not isinstance(url_candidate, str):
        raise TypeError("url_candidate must be a str")

    m = _SCHEME_REGEX.match(url_candidate)
    if m:
        return m.group(1).lower()
    return "unknown"


def detect_js_scheme(url_string: str) -> bool:
    """
    Determine if the given string contains potentially harmful URL schemes such as
    'javascript:' or 'vbscript:'. Data URLs are treated as harmful except when they
    are image data URLs (e.g., 'data:image/jpeg;base64,...'), which are ignored.

    Args:
        url_string: Input string that may contain a URL or snippet.

    Returns:
        True if a harmful scheme is present; otherwise False.
    """
    if not isinstance(url_string, str):
        raise TypeError("url_string must be a str")

    # Detect explicit script-like schemes.
    if _HARMFUL_SCHEME_RE.search(url_string):
        return True

    # Detect data: occurrences and flag as harmful unless clearly an image data URL.
    for m in _DATA_SCHEME_RE.finditer(url_string):
        tail = url_string[m.start():]
        if not _SAFE_IMAGE_DATA_START_RE.match(tail):
            return True

    return False
