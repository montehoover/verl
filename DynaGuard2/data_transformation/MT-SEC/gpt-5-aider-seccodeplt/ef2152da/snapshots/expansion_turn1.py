from typing import Any, Dict
from urllib.parse import urlparse


__all__ = ["extract_url_parts"]


def extract_url_parts(url: str) -> Dict[str, Any]:
    """
    Parse a URL string into its components.

    Returns a dictionary with:
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

    candidate = url.strip()

    # Initial parse
    parsed = urlparse(candidate)

    # Handle schemeless URLs like "example.com/path"
    if (
        not parsed.scheme
        and not parsed.netloc
        and "://" not in candidate
        and not candidate.startswith("/")
        and candidate
    ):
        parsed = urlparse(f"//{candidate}")

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
