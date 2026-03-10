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
        if url[-1] == '}' and url.count('{)') < url.count('}'):
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
