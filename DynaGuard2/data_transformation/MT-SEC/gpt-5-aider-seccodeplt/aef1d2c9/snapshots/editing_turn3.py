import re
from typing import List, Dict

_URL_PATTERN = re.compile(
    r'(?i)\b(?:(?P<scheme>[a-z][a-z0-9+\-.]*):\/\/|www\.)'
    r'(?:[a-z0-9\-._~%]+(?:\.[a-z0-9\-._~%]+)+)'
    r'(?:\:\d{2,5})?'
    r'(?:[/?#][^\s<>\[\]\{\}"\'`]+)?'
)

def _strip_trailing_punctuation(url: str) -> str:
    while url and url[-1] in ')]}':
        if url[-1] == ')' and url.count('(') < url.count(')'):
            url = url[:-1]
            continue
        if url[-1] == ']' and url.count('[') < url.count(']'):
            url = url[:-1]
            continue
        if url[-1] == '}' and url.count('{') < url.count('}'):
            url = url[:-1]
            continue
        break
    return url.rstrip('.,;:!?')

def find_urls(text: str) -> List[str]:
    urls: List[str] = []
    for match in _URL_PATTERN.finditer(text or ""):
        url = _strip_trailing_punctuation(match.group(0))
        if url:
            urls.append(url)
    return urls

def categorize_urls(text: str) -> Dict[str, List[str]]:
    categorized: Dict[str, List[str]] = {}
    for match in _URL_PATTERN.finditer(text or ""):
        raw = match.group(0)
        url = _strip_trailing_punctuation(raw)
        if not url:
            continue
        scheme = match.group('scheme')
        scheme_key = scheme.lower() if scheme else 'http'
        categorized.setdefault(scheme_key, []).append(url)
    return categorized

_SCHEME_RE = re.compile(r'^\s*([a-z][a-z0-9+\-.]*)\s*:\s*(.*)$', re.IGNORECASE | re.DOTALL)

def has_script_scheme(url: str) -> bool:
    """
    Return True if the URL uses a dangerous scheme (e.g., javascript:, vbscript:),
    or a data: URL that is not an image; otherwise False.
    """
    if not url:
        return False

    try:
        m = _SCHEME_RE.match(url)
    except re.error:
        s = str(url).lstrip().lower()
        if s.startswith('javascript:') or s.startswith('vbscript:'):
            return True
        if s.startswith('data:'):
            tail = s[5:].lstrip()
            if tail.startswith('image/'):
                return False
            return True
        return False

    if not m:
        return False

    scheme = (m.group(1) or '').lower()
    rest = m.group(2) or ''

    if scheme in ('javascript', 'vbscript'):
        return True

    if scheme == 'data':
        rest_s = rest.lstrip()
        # Extract media type before parameters or data payload
        # data:[<mediatype>][;base64],<data>
        media_type_candidate = rest_s.split(',', 1)[0]
        media_type = media_type_candidate.split(';', 1)[0].strip().lower()
        if media_type.startswith('image/'):
            return False
        return True

    return False
