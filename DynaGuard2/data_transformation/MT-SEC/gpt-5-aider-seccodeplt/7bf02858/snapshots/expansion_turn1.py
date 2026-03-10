from urllib.parse import urlparse
from typing import Dict


def extract_url_components(url: str) -> Dict[str, str]:
    """
    Extract components from a URL.

    Returns a dictionary containing:
    - scheme
    - netloc
    - path
    - params
    - query
    - fragment
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
