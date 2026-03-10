import re

# Precompiled regex to find URL-like substrings and optionally capture schemes.
_URL_PATTERN = re.compile(
    r'('
    # Scheme-based URLs (e.g., http://, https://, ftp://, custom schemes)
    r'(?P<scheme>[a-z][a-z0-9+.-]*):\/\/[^\s<>()\[\]{}"\'`]+'
    r'|'
    # www-prefixed URLs (e.g., www.example.com/path)
    r'www\.[^\s<>()\[\]{}"\'`]+'
    r'|'
    # Bare domains with optional path (e.g., example.com, sub.example.co.uk/foo)
    r'\b(?:[a-z0-9-]+\.)+[a-z]{2,}\b(?:\/[^\s<>()\[\]{}"\'`]+)?'
    r')',
    re.IGNORECASE,
)

def find_urls(text):
    """
    Scan a string and return a list of (url, scheme) tuples without strict validation.
    - url: the matched URL-like substring (with trailing punctuation trimmed)
    - scheme: the URL scheme if explicitly present (e.g., 'http', 'https'), otherwise None
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    results = []
    for m in _URL_PATTERN.finditer(text):
        url = m.group(0)
        # Trim common trailing punctuation that often follows URLs in prose.
        url = url.rstrip('.,;:!?)]}\'"')
        if not url:
            continue

        scheme = m.group('scheme')
        if scheme:
            scheme = scheme.lower()

        results.append((url, scheme))
    return results
