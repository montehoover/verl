from urllib.parse import urlparse
from typing import Dict

def extract_url_components(url: str) -> Dict[str, str]:
    """
    Parse a URL and return its components.

    Args:
        url: The URL string to parse.

    Returns:
        A dict with keys: scheme, netloc, path, params, query, fragment.
    """
    parsed = urlparse(url)
    return {
        "scheme": parsed.scheme,
        "netloc": parsed.netloc,
        "path": parsed.path,
        "params": parsed.params,
        "query": parsed.query,
        "fragment": parsed.fragment,
    }
