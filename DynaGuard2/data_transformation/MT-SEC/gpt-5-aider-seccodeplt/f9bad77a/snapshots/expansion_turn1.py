import re
import ipaddress
from urllib.parse import urlsplit


def _is_valid_scheme(scheme: str) -> bool:
    # RFC 3986: scheme = ALPHA *( ALPHA / DIGIT / "+" / "-" / "." )
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9+.\-]*", scheme))


def _is_valid_hostname(host: str) -> bool:
    # Allow localhost explicitly
    if host.lower() == "localhost":
        return True

    # Try IPv4/IPv6
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        pass

    # Convert Unicode domains to IDNA ASCII and validate label rules
    try:
        ascii_host = host.encode("idna").decode("ascii")
    except UnicodeError:
        return False

    if len(ascii_host) > 253:
        return False

    labels = ascii_host.split(".")
    for label in labels:
        if not (1 <= len(label) <= 63):
            return False
        if label.startswith("-") or label.endswith("-"):
            return False
        if not re.fullmatch(r"[A-Za-z0-9-]+", label):
            return False

    return True


def parse_and_validate_url(url: str) -> dict:
    """
    Parse and validate a URL string.

    Args:
        url: The URL to parse.

    Returns:
        A dictionary containing the parsed URL components:
        scheme, netloc, path, query, fragment, username, password, hostname, port.

    Raises:
        ValueError: If the URL is invalid.
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")
    url = url.strip()
    if not url:
        raise ValueError("URL must be a non-empty string")

    result = urlsplit(url)

    if not result.scheme:
        raise ValueError("URL must include a scheme (e.g., 'https')")
    if not _is_valid_scheme(result.scheme):
        raise ValueError(f"Invalid URL scheme: {result.scheme!r}")

    # For typical network schemes, require a netloc (host).
    # If you need to support non-network schemes (e.g., mailto), adjust as needed.
    if not result.netloc:
        raise ValueError("URL must include a network location (host)")

    # Validate port; accessing .port can raise ValueError if it's not numeric
    try:
        port = result.port
    except ValueError:
        raise ValueError("Invalid port in URL")

    if port is not None and not (1 <= port <= 65535):
        raise ValueError(f"Port out of range: {port}")

    # Validate hostname
    host = result.hostname
    if not host or not _is_valid_hostname(host):
        raise ValueError("Invalid hostname in URL")

    # Basic path validation (optional): allow empty or starting with '/'
    path = result.path or ""
    if path and not path.startswith("/"):
        # Most network URLs either have empty path or start with '/'
        raise ValueError("Invalid path in URL (must start with '/' if present)")

    return {
        "scheme": result.scheme,
        "netloc": result.netloc,
        "path": result.path,
        "query": result.query,
        "fragment": result.fragment,
        "username": result.username,
        "password": result.password,
        "hostname": result.hostname,
        "port": port,
    }
