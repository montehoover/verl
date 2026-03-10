from typing import Any, Dict, Optional
from urllib.parse import urlparse


def extract_url_parts(url: str) -> Dict[str, Any]:
    """
    Parse a URL string and return its components as a dictionary.

    Returns keys:
      - scheme: str
      - username: Optional[str]
      - password: Optional[str]
      - hostname: Optional[str]
      - port: Optional[int]
      - path: str
      - params: str
      - query: str
      - fragment: str
      - netloc: str
    """
    parsed = urlparse(url)

    return {
        "scheme": parsed.scheme,
        "username": parsed.username,
        "password": parsed.password,
        "hostname": parsed.hostname,
        "port": parsed.port,
        "path": parsed.path,
        "params": parsed.params,
        "query": parsed.query,
        "fragment": parsed.fragment,
        "netloc": parsed.netloc,
    }
