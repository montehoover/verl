import re
import ipaddress
from urllib.parse import urlsplit


_ALLOWED_SCHEMES = {"http", "https", "ws", "wss", "ftp", "ftps"}


def _is_valid_hostname(hostname: str) -> bool:
    # Accept IP addresses (IPv4/IPv6)
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        pass

    # Validate IDNA-encodable hostnames
    try:
        ascii_host = hostname.encode("idna").decode("ascii")
    except UnicodeError:
        return False

    if len(ascii_host) > 253:
        return False

    # Allow trailing dot in hostname representation for FQDNs
    if ascii_host.endswith("."):
        ascii_host = ascii_host[:-1]

    labels = ascii_host.split(".")
    label_re = re.compile(r"^[A-Za-z0-9-]{1,63}$")

    for label in labels:
        if not label:
            return False
        if not label_re.match(label):
            return False
        if label[0] == "-" or label[-1] == "-":
            return False

    return True


def validate_url(url: str) -> bool:
    """
    Validate that the given URL is well-formed.
    Returns True if valid; otherwise raises ValueError.
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")

    candidate = url.strip()
    if not candidate:
        raise ValueError("URL is empty")

    if re.search(r"\s", candidate):
        raise ValueError("URL must not contain whitespace characters")

    parsed = urlsplit(candidate)

    if not parsed.scheme:
        raise ValueError("URL must include a scheme (e.g., http, https)")
    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")

    if not parsed.netloc:
        raise ValueError("URL must include a network location (host)")

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL must include a valid hostname")

    if not _is_valid_hostname(hostname):
        raise ValueError("Invalid hostname in URL")

    # Validate port (if present)
    try:
        port = parsed.port
    except ValueError:
        raise ValueError("Invalid port in URL")
    if port is not None and not (1 <= port <= 65535):
        raise ValueError("Port out of valid range (1-65535)")

    return True
