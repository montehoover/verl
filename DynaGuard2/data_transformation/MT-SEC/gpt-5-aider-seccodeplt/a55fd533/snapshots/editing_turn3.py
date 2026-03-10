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

# Precompiled regex to find potential schemes like "javascript:", "vbscript:", etc.
_SCHEME_FINDER = re.compile(
    r'(?i)(?<![a-z0-9+.-])(?P<scheme>[a-z][a-z0-9+.-]*):'
)

# Precompiled regex to match legitimate image data URLs so they can be ignored.
_IMAGE_DATA_URL_PATTERN = re.compile(
    r'(?i)(?<![a-z0-9+.-])data:image/[a-z0-9.+-]+(?:;[a-z0-9.+-]+)*(?:;base64)?,[^\s<>()\[\]{}"\'`]+'
)

_DANGEROUS_SCHEMES = {"javascript", "vbscript"}


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


def contains_dangerous_scheme(user_input):
    """
    Return True if the input contains dangerous URL schemes (e.g., javascript:, vbscript:).
    Image data URLs (data:image/...) are ignored and do not trigger detection.
    """
    try:
        text = user_input if isinstance(user_input, str) else str(user_input)
    except Exception:
        return False

    try:
        # Collect spans of image data URLs to ignore any matches inside them.
        ignore_spans = []
        for img in _IMAGE_DATA_URL_PATTERN.finditer(text):
            ignore_spans.append((img.start(), img.end()))

        def _inside_ignored(pos):
            for s, e in ignore_spans:
                if s <= pos < e:
                    return True
            return False

        for m in _SCHEME_FINDER.finditer(text):
            if _inside_ignored(m.start()):
                continue
            scheme = m.group('scheme').lower()
            if scheme in _DANGEROUS_SCHEMES:
                return True
        return False
    except re.error:
        return False
    except Exception:
        return False
