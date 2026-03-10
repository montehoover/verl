import re
import ipaddress
from urllib.parse import urlparse


_DOMAIN_REGEX = re.compile(
    r"^(?=.{1,253}$)(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,63}$"
)


def _is_valid_hostname(hostname: str) -> bool:
    if not hostname:
        return False
    if hostname.lower() == "localhost":
        return True
    # Try IP (v4 or v6)
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        pass
    # Domain/FQDN
    return _DOMAIN_REGEX.match(hostname) is not None


def validate_and_parse_url(url: str) -> dict:
    """
    Validate the given URL and return its components.

    Args:
        url: The URL string to validate and parse.

    Returns:
        A dict with keys: scheme, domain, port, path, query, fragment, username, password.

    Raises:
        ValueError: If the URL is invalid.
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")

    candidate = url.strip()
    if not candidate:
        raise ValueError("URL must not be empty")

    parsed = urlparse(candidate)

    # Basic structural checks
    if not parsed.scheme:
        raise ValueError("URL must include a scheme (e.g., 'https')")
    if not parsed.netloc:
        raise ValueError("URL must include a network location (domain)")

    hostname = parsed.hostname  # lowercased, without brackets for IPv6
    if not _is_valid_hostname(hostname or ""):
        raise ValueError("URL has an invalid domain/host")

    # Validate port (may raise ValueError if invalid)
    try:
        port = parsed.port
    except ValueError:
        raise ValueError("URL has an invalid port")

    # Normalize path: default to "/" for network URLs
    path = parsed.path if parsed.path else "/"

    return {
        "scheme": parsed.scheme,
        "domain": hostname,
        "port": port,
        "path": path,
        "query": parsed.query,
        "fragment": parsed.fragment,
        "username": parsed.username,
        "password": parsed.password,
    }
