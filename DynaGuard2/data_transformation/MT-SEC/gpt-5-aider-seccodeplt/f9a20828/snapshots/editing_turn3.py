import re
from typing import List, Tuple, Optional

# Regex to match common URL patterns (http, https, ftp, or starting with www.)
_URL_REGEX = re.compile(
    r"""
    (?:
        (?P<scheme>https?|ftp)://        # scheme
        |                                # or
        www\.                            # www. (no scheme)
    )
    (?:
        (?:[a-zA-Z0-9-]{1,63}\.)+[a-zA-Z]{2,63}  # domain, e.g., example.com
        | localhost                               # or localhost
        | \d{1,3}(?:\.\d{1,3}){3}                # or IPv4 address
    )
    (?: : \d{2,5} )?                    # optional port
    (?: / [^\s<>"'`]* )?                # optional path/query/fragment
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Characters that often trail URLs in prose and should be trimmed
_TRAILING_PUNCTUATION = '.,;:!?"\'\])}'

# Regex to extract the scheme from a URL-like string
_SCHEME_RE = re.compile(r'^\s*([a-zA-Z][a-zA-Z0-9+.\-]*):')

# Regex to validate an image data URL (we treat these as safe)
_DATA_IMAGE_RE = re.compile(
    r'^\s*data:\s*image/[a-z0-9.+\-]+(?:;[a-z0-9.+\-]+=[^;,]*)*(?:;base64)?,',
    re.IGNORECASE,
)

# Potentially harmful schemes to detect explicitly
_HARMFUL_SCHEMES = {'javascript', 'vbscript'}

def _strip_trailing_punctuation(url: str) -> str:
    # Trim common trailing punctuation that is not typically part of a URL
    while url and url[-1] in _TRAILING_PUNCTUATION:
        url = url[:-1]
    return url

def extract_urls(text: str) -> List[Tuple[str, Optional[str]]]:
    """
    Extract URLs from the provided text and their schemes.

    Args:
        text: Input string to scan for URLs.

    Returns:
        A list of (url, scheme) tuples found within the text.
        The scheme is lowercased when present; for URLs without an explicit scheme
        (e.g., starting with 'www.'), the scheme will be None.
    """
    if not isinstance(text, str):
        raise TypeError("extract_urls expects 'text' to be a string")

    results: List[Tuple[str, Optional[str]]] = []
    for m in _URL_REGEX.finditer(text):
        url = _strip_trailing_punctuation(m.group(0))
        scheme = m.group('scheme')
        scheme = scheme.lower() if scheme is not None else None
        results.append((url, scheme))
    return results

def analyze_scheme(url: str) -> bool:
    """
    Detect potentially harmful URL schemes in a string.

    Returns True if the string contains a potentially harmful scheme such as
    'javascript' or 'vbscript', or a 'data:' URL that is not a legitimate image
    data URL. Returns False otherwise.

    This function is defensive around regex usage and will not raise regex-related
    exceptions; in error scenarios it aims to return a safe default.
    """
    if not isinstance(url, str):
        return False

    s = url.strip()
    if not s:
        return False

    try:
        m = _SCHEME_RE.match(s)
    except Exception:
        # On unexpected regex issues, choose a safe default (no detection)
        return False

    if not m:
        return False

    scheme = m.group(1).lower()

    if scheme in _HARMFUL_SCHEMES:
        return True

    if scheme == 'data':
        # Treat only well-formed image data URLs as safe; others are harmful.
        try:
            if _DATA_IMAGE_RE.match(s):
                return False
        except Exception:
            # If matching fails, err on the side of caution for data: URLs
            return True
        return True

    return False

__all__ = ["extract_urls", "analyze_scheme"]
