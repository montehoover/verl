from typing import Any, Dict, Optional
from urllib.parse import urlsplit


def extract_url_parts(url: str) -> Dict[str, Any]:
    """
    Parse a URL string and return its components in a dictionary.

    Returns a dict with the following keys:
      - scheme: str | None
      - username: str | None
      - password: str | None
      - hostname: str | None
      - port: int | None
      - path: str
      - params: str
      - query: str
      - fragment: str
    """
    if not isinstance(url, str):
        raise TypeError("url must be a string")

    parts = urlsplit(url)

    return {
        "scheme": parts.scheme or None,
        "username": parts.username,
        "password": parts.password,
        "hostname": parts.hostname,
        "port": parts.port,
        "path": parts.path,
        "params": parts.params,
        "query": parts.query,
        "fragment": parts.fragment,
    }
