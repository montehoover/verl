import re
from typing import List

# Regex to extract:
# - numbers (integers, decimals, optional thousands separators, optional exponent, optional leading sign)
# - words (Unicode letters, allowing internal apostrophes or hyphens)
# - special characters (punctuation, symbols, underscore), each as individual tokens
TOKEN_RE = re.compile(
    r"""
    (?P<number>
        [+-]?
        (?:
            (?:\d{1,3}(?:,\d{3})+|\d+)   # integer part (with optional thousands separators)
            (?:\.\d+)?                   # optional decimal part
            (?:[eE][+-]?\d+)?            # optional exponent
        )
    )
    |
    (?P<word>
        [^\W\d_]+(?:['-][^\W\d_]+)*      # Unicode letters, allowing internal ' or -
    )
    |
    (?P<special>
        _|[^\w\s]                         # underscore or any non-word, non-space char
    )
    """,
    re.UNICODE | re.VERBOSE,
)

# Regex to identify a URL scheme at the start of a string (RFC 3986)
SCHEME_RE = re.compile(r'^\s*([A-Za-z][A-Za-z0-9+.\-]*):')

# Regex to identify dangerous schemes anywhere in the string
DANGEROUS_SCHEME_RE = re.compile(
    r'(?i)(?<![A-Za-z0-9+.\-])(javascript|vbscript|data|file)\s*:',
    re.IGNORECASE
)

# Regex to recognize safe image data URLs after 'data:'
SAFE_DATA_IMAGE_RE = re.compile(
    r'^\s*image/(?:jpeg|jpg|png|gif|webp|bmp|x-icon|ico|avif)'
    r'(?:;[A-Za-z0-9.+-]+(?:=[A-Za-z0-9.+/%:-]+)?)*,',
    re.IGNORECASE
)


def extract_components(text: str) -> List[str]:
    """
    Extract a list of components (words, numbers, and special characters) from the given text.

    - Words: sequences of Unicode letters (optionally containing internal apostrophes or hyphens)
    - Numbers: integers or decimals, optionally with thousands separators, exponents, and leading +/-
    - Special characters: punctuation, symbols, underscore (each returned as a separate component)

    Whitespace is ignored and not included in the result.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return [m.group(0) for m in TOKEN_RE.finditer(text)]


def identify_url_scheme(url: str) -> str:
    """
    Identify and return the URL scheme from the given string.

    Returns the scheme in lowercase (e.g., 'http', 'https', 'javascript').
    If no scheme is found, returns 'no_scheme'.
    """
    if not isinstance(url, str):
        raise TypeError("url must be a string")
    m = SCHEME_RE.match(url)
    if m:
        return m.group(1).lower()
    return 'no_scheme'


def contains_dangerous_scheme(user_input: str) -> bool:
    """
    Check if the input contains dangerous URL schemes (e.g., 'javascript:', 'vbscript:', 'file:', 'data:'),
    while ignoring valid image data URLs such as 'data:image/jpeg;base64,...'.

    Returns True if a dangerous scheme is found (excluding safe image data URLs), otherwise False.
    """
    if not isinstance(user_input, str):
        raise TypeError("user_input must be a string")

    for m in DANGEROUS_SCHEME_RE.finditer(user_input):
        scheme = m.group(1).lower()
        if scheme == 'data':
            # Check if this is a safe image data URL (e.g., data:image/png;base64,...)
            after_colon = user_input[m.end():]
            if SAFE_DATA_IMAGE_RE.match(after_colon):
                continue  # safe image data URL, ignore
            return True
        else:
            return True

    return False
