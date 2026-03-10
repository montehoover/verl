from typing import Optional
from urllib.parse import urlparse


def is_valid_string(value: str) -> bool:
    """
    Check if the input is a non-empty string that contains no space characters.
    """
    if not isinstance(value, str):
        return False
    return bool(value) and ' ' not in value


def extract_domain(url: str) -> Optional[str]:
    """
    Extract the domain (hostname) from a URL-like string.
    Returns the domain if the string starts with http:// or https:// and has a valid hostname,
    otherwise returns None.
    """
    if not isinstance(url, str):
        return None

    s = url.strip()
    parsed = urlparse(s)

    # Ensure correct scheme
    if parsed.scheme.lower() not in ("http", "https"):
        return None

    host = parsed.hostname
    if not host or " " in host:
        return None

    return host
