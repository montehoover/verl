from urllib.parse import urlparse
from typing import Dict

def parse_image_url(url: str) -> Dict[str, str]:
    """
    Parse a URL and return its components as a dictionary.

    Keys: scheme, netloc, path, params, query, fragment
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
