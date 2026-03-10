from urllib.parse import urlparse
import ipaddress
import re
from typing import Any, Dict


_SCHEME_RE = re.compile(r'^[A-Za-z][A-Za-z0-9+.-]*$')
_NETLOC_REQUIRED_SCHEMES = {'http', 'https', 'ws', 'wss', 'ftp', 'ftps'}


def _is_valid_hostname(hostname: str) -> bool:
    # Allow trailing dot for FQDNs
    if hostname.endswith('.'):
        hostname = hostname[:-1]
    if not hostname:
        return False
    if len(hostname) > 253:
        return False
    labels = hostname.split('.')
    # Validate each label
    for label in labels:
        if not (1 <= len(label) <= 63):
            return False
        # RFC 1035 label: letters, digits, hyphens; not starting/ending with hyphen
        if not re.match(r'^[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?$', label):
            return False
    # TLD should not be all numeric
    if labels and labels[-1].isdigit():
        return False
    return True


def _validate_host(host: str) -> bool:
    # Try IP address first
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        pass
    # If not IP, validate as hostname (potentially punycode)
    return _is_valid_hostname(host)


def parse_and_validate_url(url: str) -> Dict[str, Any]:
    """
    Parse and validate a URL.

    Args:
        url: The URL string to parse.

    Returns:
        A dictionary with parsed URL components:
        scheme, netloc, path, params, query, fragment, username, password, hostname, port

    Raises:
        ValueError: If the URL is invalid.
    """
    if not isinstance(url, str):
        raise ValueError("Invalid URL: URL must be a string")

    # Trim surrounding whitespace
    url = url.strip()
    if not url:
        raise ValueError("Invalid URL: URL is empty")

    # Disallow unencoded whitespace characters inside the URL
    if any(ch.isspace() for ch in url):
        raise ValueError("Invalid URL: URL contains unencoded whitespace")

    result = urlparse(url)

    scheme = result.scheme
    if not scheme or not _SCHEME_RE.match(scheme):
        raise ValueError("Invalid URL: missing or invalid scheme")

    scheme_lower = scheme.lower()

    # Accessing .port can raise ValueError for invalid values
    try:
        port = result.port
    except ValueError:
        raise ValueError("Invalid URL: invalid port")

    hostname = result.hostname

    if scheme_lower in _NETLOC_REQUIRED_SCHEMES:
        if not result.netloc or not hostname:
            raise ValueError("Invalid URL: missing host")
        if not _validate_host(hostname):
            raise ValueError("Invalid URL: invalid host")
        if port is not None:
            if not (1 <= int(port) <= 65535):
                raise ValueError("Invalid URL: port out of range")
    else:
        # For other schemes (e.g., mailto, data, file), if a host is present, validate it
        if hostname and not _validate_host(hostname):
            raise ValueError("Invalid URL: invalid host")

    return {
        "scheme": result.scheme,
        "netloc": result.netloc,
        "path": result.path,
        "params": result.params,
        "query": result.query,
        "fragment": result.fragment,
        "username": result.username,
        "password": result.password,
        "hostname": hostname,
        "port": port,
    }
