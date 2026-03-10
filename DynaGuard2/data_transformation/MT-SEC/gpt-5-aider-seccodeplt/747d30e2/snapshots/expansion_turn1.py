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
