import re
from typing import List

URL_PATTERN = re.compile(r"""
\b
(
    (?:(?:https?|ftp)://)[^\s<>"']+       # scheme-based URLs
  |
    (?:www\.)[^\s<>"']+                   # www. URLs
  |
    (?<!@)(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,63}(?:/[^\s<>"']*)?  # bare domains with optional path, not emails
)
""", re.IGNORECASE | re.VERBOSE)

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

def find_urls(text: str) -> List[str]:
    """
    Scan the input text and return a list of URL-like substrings.
    Matches scheme URLs (http, https, ftp), www-prefixed URLs, and bare domains with optional paths.
    Trailing punctuation commonly found in prose is trimmed from the results.
    """
    if not isinstance(text, str) or not text:
        return []
    urls: List[str] = []
    for m in URL_PATTERN.finditer(text):
        candidate = m.group(0)
        cleaned = _trim_url(candidate)
        if cleaned:
            urls.append(cleaned)
    return urls
