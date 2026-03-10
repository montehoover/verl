from typing import Any, Dict, Optional
from urllib.parse import urlparse


__all__ = ["extract_url_parts", "check_internal_access"]


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


def check_internal_access(url: str) -> Optional[bool]:
    """
    Return True if the URL targets internal/private addresses such as:
      - localhost
      - 127.0.0.1
      - 10.0.0.0/8 (detected via '10.' prefix)
      - 192.168.0.0/16 (detected via '192.168.' prefix)

    Returns:
      - True if the URL is considered internal/private according to the above rules.
      - None if the URL is considered safe (not matching the internal rules).
    """
    if not isinstance(url, str):
        raise TypeError("url must be a string")

    parts = extract_url_parts(url)
    host = parts.get("hostname")
    if not host:
        return None

    host_l = host.lower()

    if host_l == "localhost":
        return True

    if host_l == "127.0.0.1":
        return True

    if host_l.startswith("10."):
        return True

    if host_l.startswith("192.168."):
        return True

    return None
