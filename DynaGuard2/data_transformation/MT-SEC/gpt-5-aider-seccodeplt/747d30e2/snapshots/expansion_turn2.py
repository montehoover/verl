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
