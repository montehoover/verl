from urllib.parse import urlparse
from typing import Tuple


def extract_url_parts(url: str) -> Tuple[str, str, str, str, str, str]:
    """
    Extract components from a URL.

    Returns a tuple: (scheme, netloc, path, params, query, fragment)
    """
    parsed = urlparse(url)
    return (
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        parsed.params,
        parsed.query,
        parsed.fragment,
    )
