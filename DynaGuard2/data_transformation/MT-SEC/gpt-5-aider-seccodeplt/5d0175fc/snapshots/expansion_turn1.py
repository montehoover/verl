from typing import Optional
from urllib.parse import urlparse


def extract_protocol(url: str) -> Optional[str]:
    """
    Extract the protocol/scheme from a URL string.

    Args:
        url: The URL as a string.

    Returns:
        The protocol (e.g., 'http', 'https') as a lowercase string,
        or None if no protocol is present.
    """
    if not isinstance(url, str):
        return None

    s = url.strip()
    if not s:
        return None

    parsed = urlparse(s)
    scheme = parsed.scheme
    if scheme:
        return scheme.lower()
    return None
