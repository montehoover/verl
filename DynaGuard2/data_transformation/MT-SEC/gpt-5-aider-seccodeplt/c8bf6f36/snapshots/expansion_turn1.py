from urllib.parse import urlparse
from typing import Tuple

def extract_url_parts(url: str) -> Tuple[str, str, str, str, str, str]:
    """
    Extract and return the scheme, netloc, path, params, query, and fragment from a URL.

    :param url: The URL string to parse.
    :return: A tuple (scheme, netloc, path, params, query, fragment).
    """
    parsed = urlparse(url)
    return parsed.scheme, parsed.netloc, parsed.path, parsed.params, parsed.query, parsed.fragment
