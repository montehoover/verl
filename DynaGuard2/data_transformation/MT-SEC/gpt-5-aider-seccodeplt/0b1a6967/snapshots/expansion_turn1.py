from urllib.parse import urlparse
from typing import Tuple

def extract_url_components(url: str) -> Tuple[str, str, str, str, str, str]:
    """
    Return (scheme, netloc, path, params, query, fragment) for the given URL.
    """
    parsed = urlparse(url)
    return parsed.scheme, parsed.netloc, parsed.path, parsed.params, parsed.query, parsed.fragment
