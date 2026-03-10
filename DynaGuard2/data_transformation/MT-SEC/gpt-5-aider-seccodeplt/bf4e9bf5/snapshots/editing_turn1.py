from urllib.parse import urlparse
from typing import Dict, Optional


def extract_url_components(url: str) -> Dict[str, Optional[str]]:
    """
    Extract components from a URL string.

    Returns a dictionary with:
      - scheme: URL scheme (e.g., 'http', 'https')
      - domain: hostname without userinfo or port (e.g., 'example.com')
      - path: path component (e.g., '/some/path')
    """
    parsed = urlparse(url)
    return {
        "scheme": parsed.scheme or "",
        "domain": parsed.hostname,
        "path": parsed.path or "",
    }
