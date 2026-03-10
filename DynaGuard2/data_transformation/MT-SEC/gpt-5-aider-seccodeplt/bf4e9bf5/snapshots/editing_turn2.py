from urllib.parse import urlparse
from typing import Dict, Optional, Iterable


def extract_url_components(url: str) -> Dict[str, Optional[str]]:
    """
    Extract components from a URL string.

    Returns a dictionary with:
      - scheme: URL scheme (e.g., 'http', 'https')
      - domain: hostname without userinfo or port (e.g., 'example.com')
      - path: path component (e.g., '/some/path')
    """
    parsed = urlparse(url)
    return {
        "scheme": parsed.scheme or "",
        "domain": parsed.hostname,
        "path": parsed.path or "",
    }


def check_url_validity(url: str, allowed_domains: Iterable[str]) -> bool:
    """
    Check whether a URL is valid based on:
      - The scheme must be 'http' or 'https'
      - The domain (hostname) must be in the provided allowed_domains list

    Args:
        url: The URL string to validate.
        allowed_domains: An iterable of allowed domain names. Matching is done
                         case-insensitively against the URL's hostname.

    Returns:
        True if the URL meets the criteria, otherwise False.
    """
    parsed = urlparse(url)

    # Validate scheme
    if parsed.scheme.lower() not in {"http", "https"}:
        return False

    # Validate hostname presence
    hostname = parsed.hostname
    if not hostname:
        return False

    # Compare case-insensitively to allowed domains
    allowed_set = {d.lower().rstrip(".") for d in allowed_domains}
    return hostname.lower().rstrip(".") in allowed_set
