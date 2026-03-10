from urllib.parse import urlparse
from typing import Any, Dict, Optional


def extract_url_parts(url: str) -> Dict[str, Optional[Any]]:
    """
    Parse a URL string and return its components as a dictionary.

    Returns dictionary keys:
      - scheme
      - netloc
      - path
      - params
      - query
      - fragment
      - username
      - password
      - hostname
      - port
    """
    if not isinstance(url, str):
        raise TypeError("url must be a string")

    parsed = urlparse(url)
    return {
        "scheme": parsed.scheme,
        "netloc": parsed.netloc,
        "path": parsed.path,
        "params": parsed.params,
        "query": parsed.query,
        "fragment": parsed.fragment,
        "username": parsed.username,
        "password": parsed.password,
        "hostname": parsed.hostname,
        "port": parsed.port,
    }
