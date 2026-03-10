import re
from typing import List

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
    re.VERBOSE,
)

# Characters that often trail URLs in prose and should be trimmed
_TRAILING_PUNCTUATION = '.,;:!?"\'\])}'

def _strip_trailing_punctuation(url: str) -> str:
    # Trim common trailing punctuation that is not typically part of a URL
    while url and url[-1] in _TRAILING_PUNCTUATION:
        url = url[:-1]
    return url

def extract_urls(text: str) -> List[str]:
    """
    Extract URLs from the provided text.

    Args:
        text: Input string to scan for URLs.

    Returns:
        A list of URLs found within the text.
    """
    if not isinstance(text, str):
        raise TypeError("extract_urls expects 'text' to be a string")

    urls = []
    for m in _URL_REGEX.finditer(text):
        url = m.group(0)
        url = _strip_trailing_punctuation(url)
        urls.append(url)
    return urls

__all__ = ["extract_urls"]
