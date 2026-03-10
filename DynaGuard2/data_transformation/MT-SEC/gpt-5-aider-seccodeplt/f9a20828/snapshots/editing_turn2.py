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

__all__ = ["extract_urls"]
