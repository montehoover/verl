from urllib.parse import urlparse
from typing import Tuple

def extract_url_components(url: str) -> Tuple[str, str, str, str, str, str]:
    """
    Extract components from a URL.

    Returns:
        A tuple of (scheme, netloc, path, params, query, fragment).
    """
    parsed = urlparse(url)
    return parsed.scheme, parsed.netloc, parsed.path, parsed.params, parsed.query, parsed.fragment
