from typing import Any, Dict, Optional
from urllib.parse import urlparse


def extract_url_parts(url: str) -> Dict[str, Optional[Any]]:
    """
    Parse a URL string and return its components in a dictionary.

    Components included:
    - scheme
    - username
    - password
    - hostname
    - port
    - path
    - params
    - query
    - fragment
    - netloc

    Args:
        url: The URL string to parse.

    Returns:
        A dictionary containing the parsed components.
    """
    parsed = urlparse(url)

    return {
        "scheme": parsed.scheme or None,
        "username": parsed.username,
        "password": parsed.password,
        "hostname": parsed.hostname,
        "port": parsed.port,
        "path": parsed.path or None,
        "params": parsed.params or None,
        "query": parsed.query or None,
        "fragment": parsed.fragment or None,
        "netloc": parsed.netloc or None,
    }
