from urllib.parse import urlparse
from typing import Tuple

VALID_REDIRECT_DOMAINS = ["trusted-redirect.com", "partner.com"]

def extract_url_components(url: str) -> Tuple[str, str, str, str, str, str]:
    """
    Extract components from a URL.

    Returns:
        A tuple of (scheme, netloc, path, params, query, fragment).
    """
    parsed = urlparse(url)
    return parsed.scheme, parsed.netloc, parsed.path, parsed.params, parsed.query, parsed.fragment

def is_valid_domain(url: str) -> bool:
    """
    Check if the URL's domain is in the approved list of domains.

    Args:
        url: The URL to check.

    Returns:
        True if the domain is approved, False otherwise.
    """
    parsed = urlparse(url)
    hostname = parsed.hostname  # Already lowercased by urlparse
    if not hostname:
        return False
    hostname = hostname.rstrip(".")
    return hostname in VALID_REDIRECT_DOMAINS
