import re
import ipaddress
from urllib.parse import urlsplit
from typing import Dict


_HOSTNAME_REGEX = re.compile(
    r"^(?=.{1,253}$)(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)*[A-Za-z0-9]"
    r"(?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$"
)


def _is_valid_hostname(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        ascii_host = host.encode("idna").decode("ascii")
    except Exception:
        return False
    return bool(_HOSTNAME_REGEX.match(ascii_host))


def validate_and_parse_url(url: str) -> Dict[str, str]:
    """
    Validate and parse a URL string.

    Returns a dict with:
      - scheme
      - domain (hostname without port or credentials)
      - path

    Raises:
        ValueError: if the URL is invalid.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("URL must be a non-empty string.")

    url = url.strip()
    parsed = urlsplit(url)

    if not parsed.scheme:
        raise ValueError("URL is missing a scheme (e.g., 'https').")

    if not parsed.netloc:
        # Handles cases like 'http:/path' or 'http://'
        raise ValueError("URL is missing a domain.")

    if any(ch.isspace() for ch in parsed.netloc):
        raise ValueError("Domain must not contain whitespace.")

    host = parsed.hostname
    if not host:
        raise ValueError("URL contains an invalid host.")

    # Validate host as IP or hostname
    try:
        ipaddress.ip_address(host)
        host_valid = True
    except ValueError:
        host_valid = _is_valid_hostname(host)

    if not host_valid:
        raise ValueError("URL contains an invalid domain or IP address.")

    path = parsed.path or "/"

    return {
        "scheme": parsed.scheme,
        "domain": host,
        "path": path,
    }
