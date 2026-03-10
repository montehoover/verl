import re
from typing import List

_URL_REGEX = re.compile(
    r"""
    \b
    (                                   # Capture the full URL
      (?:                               # Scheme-based URLs
        (?:https?://|ftp://)
        (?:                             # Host
          localhost
          |
          \d{1,3}(?:\.\d{1,3}){3}
          |
          (?:[a-z0-9\-._~%]+\.)+[a-z]{2,}
        )
      )
      |
      (?:                               # www.-style URLs without scheme
        www\.(?:[a-z0-9\-._~%]+\.)+[a-z]{2,}
      )
    )
    (?:\:\d{2,5})?                      # Optional port
    (?:[/?#][^\s<>"'(){}\[\]]*)?        # Optional path/query/fragment
    """,
    re.IGNORECASE | re.VERBOSE,
)

def find_urls(text: str) -> List[str]:
    """
    Find URL-like patterns within a string.

    Args:
        text: Input text to search.

    Returns:
        List of URL-like strings found.
    """
    return _URL_REGEX.findall(text)

_SCHEME_REGEX = re.compile(r'^\s*([A-Za-z][A-Za-z0-9+.\-]*)\:')

def extract_url_scheme(url: str) -> str:
    """
    Extract the scheme from a URL-like string.

    Examples of returned schemes: 'http', 'https', 'mailto', 'javascript', etc.
    If no scheme is present, returns 'no_scheme'.
    """
    match = _SCHEME_REGEX.match(url)
    if not match:
        return 'no_scheme'

    scheme = match.group(1).lower()

    # Avoid false positives for Windows drive letters like "C:\path"
    if len(scheme) == 1 and re.match(r'^\s*[A-Za-z]:[\\/]', url):
        return 'no_scheme'

    return scheme

# Detect potentially malicious script schemes like javascript:, jscript:, vbscript:, etc.
_MALICIOUS_SCHEME_RE = re.compile(
    r'(?is)(?:^|[^a-z0-9+.\-])(?:java\s*script|jscript|vbscript|livescript|mocha)\s*:',
    re.IGNORECASE,
)

# Detect data image URLs to explicitly consider them non-malicious (e.g., data:image/jpeg;base64,...)
_DATA_IMAGE_PREFIX_RE = re.compile(
    r'(?i)^\s*data\s*:\s*image\/[a-z0-9.+-]+(?:;base64)?\s*,'
)

def is_javascript_scheme(s: str) -> bool:
    """
    Return True if the string contains a potentially malicious script scheme
    like 'javascript:', 'jscript:', 'vbscript:', etc.
    Image data URLs like 'data:image/jpeg;base64,...' are considered safe.
    """
    if _MALICIOUS_SCHEME_RE.search(s):
        return True

    # Explicitly treat image data URLs as non-malicious
    if _DATA_IMAGE_PREFIX_RE.match(s):
        return False

    return False
