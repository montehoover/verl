from urllib.parse import urlparse
from typing import Dict

def parse_image_url(url: str) -> Dict[str, str]:
    """
    Parse an image URL and return its components: scheme, netloc, path, params, query, and fragment.
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
