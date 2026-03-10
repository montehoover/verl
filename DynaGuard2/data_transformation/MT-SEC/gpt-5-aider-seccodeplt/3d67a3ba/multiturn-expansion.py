from typing import Optional
from urllib.parse import urlparse
import re


# Precompiled URL validation regex (basic structure for http/https URLs)
_URL_REGEX = re.compile(
    r'^(?:(?:http|https)://)'                  # scheme
    r'(?:[^\s/@]+@)?'                          # optional userinfo
    r'(?:'                                     # host
      r'localhost|'                            # localhost
      r'(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,}|'  # domain
      r'(?:\d{1,3}\.){3}\d{1,3}'               # IPv4 (basic, not range-validated)
    r')'
    r'(?::\d{2,5})?'                           # optional port
    r'(?:[/?#][^\s]*)?'                        # optional path/query/fragment
    r'$', re.IGNORECASE
)


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


def check_url_format(link: str) -> bool:
    """
    Return True if the given string is a correctly formatted http/https URL based on a regex,
    or False otherwise. Never raises exceptions.
    """
    try:
        if not isinstance(link, str):
            return False
        s = link.strip()
        if not s:
            return False
        return _URL_REGEX.match(s) is not None
    except Exception:
        return False
