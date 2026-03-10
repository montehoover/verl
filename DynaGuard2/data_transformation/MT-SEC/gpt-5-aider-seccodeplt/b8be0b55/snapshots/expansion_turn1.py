from urllib.parse import urlparse
from typing import Dict


def extract_url_parts(url: str) -> Dict[str, str]:
    """
    Extract components from a URL.

    Returns a dictionary with keys: scheme, netloc, path, params, query, fragment.
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
