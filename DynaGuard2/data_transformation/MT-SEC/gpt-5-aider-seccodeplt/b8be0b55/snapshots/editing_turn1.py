from urllib.parse import urlparse
from typing import Dict

def extract_url_parts(url: str) -> Dict[str, str]:
    """
    Extracts parts of a URL and returns them in a dictionary.

    Returns:
        dict with keys: scheme, netloc, path
    """
    if not isinstance(url, str):
        raise TypeError("url must be a string")

    parsed = urlparse(url)
    return {
        "scheme": parsed.scheme,
        "netloc": parsed.netloc,
        "path": parsed.path,
    }
