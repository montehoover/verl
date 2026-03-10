from urllib.parse import urlparse
from typing import Any, Dict, Optional


def extract_url_parts(url: str) -> Dict[str, Optional[Any]]:
    """
    Parse a URL string and return its components as a dictionary.

    Returned keys:
      - scheme
      - username
      - password
      - hostname
      - port
      - path
      - query
      - fragment
    """
    if not isinstance(url, str):
        raise TypeError("url must be a string")

    parsed = urlparse(url)

    # Handle schemeless URLs like "example.com/path"
    if not parsed.scheme and not parsed.netloc and parsed.path:
        parsed = urlparse("//" + url)

    return {
        "scheme": parsed.scheme or None,
        "username": parsed.username or None,
        "password": parsed.password or None,
        "hostname": parsed.hostname or None,
        "port": parsed.port,
        "path": parsed.path or None,
        "query": parsed.query or None,
        "fragment": parsed.fragment or None,
    }
