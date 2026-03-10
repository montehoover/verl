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

# Precompiled regexes for harmful scheme detection
_HARMFUL_SCHEME_RE = re.compile(r'(?i)\b(?:javascript|vbscript|file)\s*:')
_DATA_SCHEME_RE = re.compile(r'(?i)\bdata\s*:\s*([^;,]*)')

_ALLOWED_IMAGE_MIME = {
    'image/png',
    'image/jpeg',
    'image/jpg',
    'image/gif',
    'image/webp',
    'image/bmp',
    'image/x-icon',
    'image/vnd.microsoft.icon',
    'image/tiff',
    'image/avif',
    'image/pjpeg',
}

def has_script_scheme(url: str) -> bool:
    """
    Return True if the string contains a potentially harmful URL scheme.
    Harmful schemes include: javascript:, vbscript:, file:, and data: (except for common image MIME types).
    Valid image data URLs such as 'data:image/jpeg;base64,...' are ignored (treated as not harmful).
    """
    # Immediate harmful schemes like javascript:, vbscript:, file:
    if _HARMFUL_SCHEME_RE.search(url):
        return True

    # data: URLs are harmful unless they are clearly common image types
    for m in _DATA_SCHEME_RE.finditer(url):
        mime = (m.group(1) or '').strip().lower()
        # Extract only the MIME type part before any parameters (e.g., ;base64) or subsequent whitespace
        if mime:
            # If there are extra parameters attached without being separated yet, cut at first space if any
            mime = re.split(r'[;\s]', mime, maxsplit=1)[0]
        # If no MIME specified or not an allowed image MIME => harmful
        if not mime or mime not in _ALLOWED_IMAGE_MIME:
            return True

    return False
