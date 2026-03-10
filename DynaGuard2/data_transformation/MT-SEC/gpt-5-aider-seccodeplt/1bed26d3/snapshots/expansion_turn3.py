import re

_TOKEN_RE = re.compile(
    r"""
    (
        [^\W\d_]+(?:[-'][^\W\d_]+)*      # words (Unicode letters, optional hyphen/apostrophe)
      | [+-]?(?:\d+(?:\.\d+)?|\.\d+)     # numbers (integers or decimals, optional sign)
      | [^\w\s]|_                        # symbols (non-word, non-space, or underscore)
    )
    """,
    re.UNICODE | re.VERBOSE,
)

_URL_SCHEME_RE = re.compile(
    r'^\s*(?![A-Za-z]:[\\/])([A-Za-z][A-Za-z0-9+.-]*):',
    re.UNICODE | re.IGNORECASE,
)

_SAFE_DATA_IMAGE_RE = re.compile(
    r'^\s*data\s*:\s*image/(?:jpeg|jpg|png|gif|bmp|webp|avif|x-icon|vnd\.microsoft\.icon|ico)(?:\s*;\s*base64)?\s*,',
    re.IGNORECASE | re.UNICODE,
)

def extract_components(text: str):
    return [m.group(0) for m in _TOKEN_RE.finditer(text)]

def identify_url_scheme(url: str):
    m = _URL_SCHEME_RE.match(url)
    if m:
        return m.group(1).lower()
    return 'no_scheme'

def check_javascript_scheme(url_str: str) -> bool:
    """
    Return True if the input string appears to use a potentially harmful URL scheme,
    such as javascript:, vbscript:, or data: (excluding safe image data URLs).
    """
    s = url_str.strip()

    # Explicitly allow common safe image data URLs (e.g., data:image/jpeg;base64,...)
    if _SAFE_DATA_IMAGE_RE.match(s):
        return False

    m = _URL_SCHEME_RE.match(s)
    if not m:
        return False

    scheme = m.group(1).lower()

    if scheme in ('javascript', 'vbscript'):
        return True

    if scheme == 'data':
        # Non-image (or unrecognized) data URLs are potentially harmful
        return True

    return False
