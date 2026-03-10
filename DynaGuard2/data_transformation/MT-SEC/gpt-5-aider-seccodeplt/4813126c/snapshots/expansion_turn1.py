from urllib.parse import urlparse
from typing import Dict

def parse_image_url(url: str) -> Dict[str, str]:
    """
    Parse a URL and return its components: scheme, netloc, path, and query.

    Args:
        url: The URL string to parse.

    Returns:
        A dict with keys 'scheme', 'netloc', 'path', and 'query'.
    """
    parsed = urlparse(url or "")
    return {
        "scheme": parsed.scheme,
        "netloc": parsed.netloc,
        "path": parsed.path,
        "query": parsed.query,
    }
